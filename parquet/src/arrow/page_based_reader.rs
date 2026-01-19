// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Page-based Parquet reader for two-stage loading.
//!
//! This module provides [`RowGroupPageBasedReader`] which accepts pre-decompressed pages
//! and produces Arrow [`RecordBatch`]es. This enables external parallel decompression
//! of Parquet pages.
//!
//! # Example
//!
//! ```ignore
//! use parquet::arrow::page_based_reader::RowGroupPageBasedReader;
//! use parquet::file::page_index::page_metadata::RowGroupPageIndex;
//! use parquet::column::page::Page;
//!
//! // Load page metadata
//! let page_index = RowGroupPageIndex::load(&file, metadata.clone(), 0)?;
//!
//! // Create reader from page index
//! let mut reader = RowGroupPageBasedReader::try_new(&page_index, batch_size)?;
//!
//! // Push pages in any order
//! reader.push_page(1, 0, decompressed_page)?;  // column 1, page 0
//! reader.push_page(0, 0, decompressed_page)?;  // column 0, page 0
//! reader.push_page(0, 1, decompressed_page)?;  // column 0, page 1
//!
//! // Read batches
//! while let Some(batch) = reader.read_record_batch()? {
//!     // process batch
//! }
//! ```

use std::cell::UnsafeCell;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;

use crate::arrow::array_reader::{ArrayReader, ArrayReaderBuilder, RowGroups};
use crate::arrow::arrow_reader::metrics::ArrowReaderMetrics;
use crate::arrow::schema::{ParquetField, parquet_to_arrow_schema_and_fields};
use crate::arrow::ProjectionMask;
use crate::column::page::{Page, PageIterator, PageMetadata, PageReader};
use crate::errors::{ParquetError, Result};
use crate::file::metadata::{ParquetMetaData, RowGroupMetaData};
use crate::file::page_index::page_metadata::RowGroupPageIndex;

/// Indexed page storage for out-of-order page pushing.
///
/// Pages are stored in fixed slots that can be filled in any order.
/// Pages are consumed sequentially starting from index 0.
///
/// # Safety
/// Access is synchronized through `&mut self` on the reader.
/// The Rust borrow checker ensures push and read cannot happen concurrently.
struct IndexedPageSlots {
    inner: UnsafeCell<IndexedPageSlotsInner>,
}

struct IndexedPageSlotsInner {
    /// Pages stored by index. `None` means not yet pushed.
    pages: Vec<Option<Page>>,
    /// Next page index to consume.
    next_read_idx: usize,
}

impl IndexedPageSlots {
    fn new(num_pages: usize) -> Self {
        Self {
            inner: UnsafeCell::new(IndexedPageSlotsInner {
                pages: vec![None; num_pages],
                next_read_idx: 0,
            }),
        }
    }

    /// Set a page at a specific index.
    ///
    /// # Safety
    /// Caller must ensure exclusive access.
    #[inline]
    unsafe fn set(&self, idx: usize, page: Page) {
        let inner = &mut *self.inner.get();
        if idx < inner.pages.len() {
            inner.pages[idx] = Some(page);
        }
    }

    /// Take the next page in sequence.
    ///
    /// # Safety
    /// Caller must ensure exclusive access.
    #[inline]
    unsafe fn take_next(&self) -> Option<Page> {
        let inner = &mut *self.inner.get();
        if inner.next_read_idx < inner.pages.len() {
            let page = inner.pages[inner.next_read_idx].take();
            if page.is_some() {
                inner.next_read_idx += 1;
            }
            page
        } else {
            None
        }
    }

    /// Peek at the next page without consuming.
    ///
    /// # Safety
    /// Caller must ensure exclusive access.
    #[inline]
    unsafe fn peek_next(&self) -> Option<&Page> {
        let inner = &*self.inner.get();
        if inner.next_read_idx < inner.pages.len() {
            inner.pages[inner.next_read_idx].as_ref()
        } else {
            None
        }
    }

    /// Get expected total pages.
    #[inline]
    fn expected_count(&self) -> usize {
        // Safe since the length is immutable after creation
        unsafe { (*self.inner.get()).pages.len() }
    }
}

// SAFETY: Access is synchronized through &mut self on RowGroupPageBasedReader.
unsafe impl Send for IndexedPageSlots {}
unsafe impl Sync for IndexedPageSlots {}

type SharedIndexedSlots = Arc<IndexedPageSlots>;

/// A stateful reader for a single row group that accepts pre-decompressed pages
/// and produces RecordBatches.
///
/// Pages can be pushed in any order using `push_page(column_idx, page_idx, page)`.
/// Pages are consumed sequentially per column as batches are read.
///
/// This implementation is lock-free - it uses `UnsafeCell` internally with safety
/// guaranteed by the Rust borrow checker (both `push_page` and `read_record_batch`
/// require `&mut self`).
pub struct RowGroupPageBasedReader {
    /// Parquet metadata
    metadata: Arc<ParquetMetaData>,
    /// Row group index
    row_group_idx: usize,
    /// Arrow schema
    schema: SchemaRef,
    /// Parquet field structure for building array readers
    fields: Option<Arc<ParquetField>>,
    /// Batch size for reading
    batch_size: usize,
    /// Projection mask
    projection: ProjectionMask,
    /// Page slots for each column
    column_pages: Vec<SharedIndexedSlots>,
    /// Current array reader (lazily built)
    array_reader: Option<Box<dyn ArrayReader>>,
    /// Total values pushed to each column (used to calculate available values)
    values_pushed: Vec<usize>,
    /// Number of rows read from this row group
    rows_read: usize,
    /// Total rows in this row group
    total_rows: usize,
}

impl std::fmt::Debug for RowGroupPageBasedReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowGroupPageBasedReader")
            .field("row_group_idx", &self.row_group_idx)
            .field("batch_size", &self.batch_size)
            .field("rows_read", &self.rows_read)
            .field("total_rows", &self.total_rows)
            .field("num_columns", &self.column_pages.len())
            .finish()
    }
}

impl RowGroupPageBasedReader {
    /// Create a new reader from a `RowGroupPageIndex`.
    ///
    /// # Arguments
    /// * `page_index` - Page metadata for the row group
    /// * `batch_size` - Number of rows per RecordBatch
    pub fn try_new(page_index: &RowGroupPageIndex, batch_size: usize) -> Result<Self> {
        Self::try_new_with_projection(page_index, batch_size, ProjectionMask::all())
    }

    /// Create a new reader from a `RowGroupPageIndex` with a projection mask.
    ///
    /// # Arguments
    /// * `page_index` - Page metadata for the row group
    /// * `batch_size` - Number of rows per RecordBatch
    /// * `projection` - Which columns to read
    pub fn try_new_with_projection(
        page_index: &RowGroupPageIndex,
        batch_size: usize,
        projection: ProjectionMask,
    ) -> Result<Self> {
        let metadata = page_index.metadata().clone();
        let row_group_idx = page_index.row_group_idx();

        let key_value_metadata = metadata.file_metadata().key_value_metadata();
        let (schema, fields) = parquet_to_arrow_schema_and_fields(
            metadata.file_metadata().schema_descr(),
            projection.clone(),
            key_value_metadata,
            &[],
        )?;
        let fields = fields.map(Arc::new);

        let total_rows = page_index.num_rows();
        let num_columns = page_index.num_columns();
        let pages_per_column = page_index.pages_per_column();

        // Create indexed slots for each column
        let column_pages: Vec<SharedIndexedSlots> = pages_per_column
            .iter()
            .map(|&count| Arc::new(IndexedPageSlots::new(count)))
            .collect();

        Ok(Self {
            metadata,
            row_group_idx,
            schema: Arc::new(schema),
            fields,
            batch_size,
            projection,
            column_pages,
            array_reader: None,
            values_pushed: vec![0; num_columns],
            rows_read: 0,
            total_rows,
        })
    }

    /// Get the Arrow schema for this reader.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Get the Parquet metadata.
    pub fn metadata(&self) -> &Arc<ParquetMetaData> {
        &self.metadata
    }

    /// Get the row group index.
    pub fn row_group_idx(&self) -> usize {
        self.row_group_idx
    }

    /// Get the number of columns.
    pub fn num_columns(&self) -> usize {
        self.column_pages.len()
    }

    /// Push a pre-decompressed page for a column.
    ///
    /// Pages can be pushed in any order. The reader will consume them
    /// sequentially per column as batches are read.
    ///
    /// # Arguments
    /// * `column_idx` - The leaf column index
    /// * `page_idx` - The page index within this column (0 = first page, may be dictionary)
    /// * `page` - The pre-decompressed page
    pub fn push_page(
        &mut self,
        column_idx: usize,
        page_idx: usize,
        page: Page,
    ) -> Result<()> {
        if column_idx >= self.column_pages.len() {
            return Err(ParquetError::General(format!(
                "Column index {} out of bounds ({})",
                column_idx,
                self.column_pages.len()
            )));
        }

        let slots = &self.column_pages[column_idx];
        if page_idx >= slots.expected_count() {
            return Err(ParquetError::General(format!(
                "Page index {} out of bounds for column {} (expected {})",
                page_idx,
                column_idx,
                slots.expected_count()
            )));
        }

        // Track values pushed (for non-nested types, values == rows)
        let num_values = match &page {
            Page::DataPage { num_values, .. } => *num_values as usize,
            Page::DataPageV2 { num_values, .. } => *num_values as usize,
            Page::DictionaryPage { .. } => 0, // Dictionary pages don't contribute rows
        };
        self.values_pushed[column_idx] += num_values;

        // SAFETY: We have &mut self, so no other access is possible
        unsafe {
            slots.set(page_idx, page);
        }

        Ok(())
    }

    /// Read the next RecordBatch.
    ///
    /// Before reading, this checks that all columns have enough values
    /// in their page slots. If any column has insufficient data, an error is
    /// returned - push more pages for those columns and retry.
    ///
    /// Returns `Ok(Some(batch))` if a batch was read successfully.
    /// Returns `Ok(None)` if there are no more rows in this row group.
    /// Returns `Err` if required pages are not yet available - push missing pages and retry.
    pub fn read_record_batch(&mut self) -> Result<Option<RecordBatch>> {
        // Check if we've read all rows
        if self.rows_read >= self.total_rows {
            return Ok(None);
        }

        // Build array reader if needed
        if self.array_reader.is_none() {
            self.build_array_reader()?;
        }

        // Calculate how many rows to read
        let remaining = self.total_rows - self.rows_read;
        let to_read = std::cmp::min(self.batch_size, remaining);

        // Pre-check: ensure all columns have enough values
        self.check_columns_have_values(to_read)?;

        let array_reader = self
            .array_reader
            .as_mut()
            .expect("array_reader should be set");

        // Read records
        let read = array_reader.read_records(to_read)?;

        if read == 0 {
            return Err(ParquetError::General(format!(
                "No data read: {} rows remaining but columns returned no records",
                remaining
            )));
        }

        self.rows_read += read;

        // Consume and build record batch
        let array = array_reader.consume_batch()?;
        let struct_array = array
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .ok_or_else(|| {
                ParquetError::General("Expected StructArray from array reader".to_string())
            })?;

        let batch = RecordBatch::from(struct_array);
        Ok(Some(batch))
    }

    /// Check that all columns have enough values for the requested read.
    fn check_columns_have_values(&self, needed: usize) -> Result<()> {
        let mut insufficient_columns = Vec::new();

        for col_idx in 0..self.column_pages.len() {
            let pushed = self.values_pushed[col_idx];
            let available = pushed.saturating_sub(self.rows_read);
            if available < needed {
                insufficient_columns.push((col_idx, available));
            }
        }

        if !insufficient_columns.is_empty() {
            let no_data: Vec<_> = insufficient_columns
                .iter()
                .filter(|(_, avail)| *avail == 0)
                .map(|(idx, _)| *idx)
                .collect();

            if !no_data.is_empty() {
                return Err(ParquetError::General(format!(
                    "No values available for columns {:?}. Push pages before reading.",
                    no_data
                )));
            }

            return Err(ParquetError::General(format!(
                "Insufficient data for batch size {}. Columns with insufficient values: {:?}. \
                 Push more pages or reduce batch size.",
                needed, insufficient_columns
            )));
        }

        Ok(())
    }

    /// Check if there are more rows to read.
    pub fn has_more_rows(&self) -> bool {
        self.rows_read < self.total_rows
    }

    /// Get the number of rows remaining.
    pub fn rows_remaining(&self) -> usize {
        self.total_rows.saturating_sub(self.rows_read)
    }

    /// Build the array reader.
    fn build_array_reader(&mut self) -> Result<()> {
        let row_group = IndexedPageRowGroup {
            metadata: self.metadata.clone(),
            row_group_idx: self.row_group_idx,
            column_slots: self.column_pages.clone(),
        };

        let metrics = ArrowReaderMetrics::Disabled;
        let builder = ArrayReaderBuilder::new(&row_group, &metrics);
        let array_reader = builder.build_array_reader(self.fields.as_deref(), &self.projection)?;

        self.array_reader = Some(array_reader);
        Ok(())
    }
}

/// A RowGroups implementation backed by indexed page slots.
struct IndexedPageRowGroup {
    metadata: Arc<ParquetMetaData>,
    row_group_idx: usize,
    column_slots: Vec<SharedIndexedSlots>,
}

impl RowGroups for IndexedPageRowGroup {
    fn num_rows(&self) -> usize {
        self.metadata.row_group(self.row_group_idx).num_rows() as usize
    }

    fn column_chunks(&self, i: usize) -> Result<Box<dyn PageIterator>> {
        let slots = self.column_slots.get(i).cloned().ok_or_else(|| {
            ParquetError::General(format!("Column index {} out of bounds", i))
        })?;

        let page_reader = IndexedPageReader::new(slots);
        Ok(Box::new(SinglePageIterator {
            reader: Some(Ok(Box::new(page_reader))),
        }))
    }

    fn row_groups(&self) -> Box<dyn Iterator<Item = &RowGroupMetaData> + '_> {
        Box::new(std::iter::once(self.metadata.row_group(self.row_group_idx)))
    }

    fn metadata(&self) -> &ParquetMetaData {
        &self.metadata
    }
}

/// A PageReader that reads from indexed page slots.
struct IndexedPageReader {
    slots: SharedIndexedSlots,
}

impl IndexedPageReader {
    fn new(slots: SharedIndexedSlots) -> Self {
        Self { slots }
    }
}

impl PageReader for IndexedPageReader {
    fn get_next_page(&mut self) -> Result<Option<Page>> {
        // SAFETY: Access is synchronized through &mut self on RowGroupPageBasedReader.
        Ok(unsafe { self.slots.take_next() })
    }

    fn peek_next_page(&mut self) -> Result<Option<PageMetadata>> {
        // SAFETY: Same as above
        let page = unsafe { self.slots.peek_next() };
        Ok(page.map(page_to_metadata))
    }

    fn skip_next_page(&mut self) -> Result<()> {
        // SAFETY: Same as above
        unsafe { self.slots.take_next() };
        Ok(())
    }
}

impl Iterator for IndexedPageReader {
    type Item = Result<Page>;

    fn next(&mut self) -> Option<Self::Item> {
        self.get_next_page().transpose()
    }
}

/// Convert a Page to PageMetadata.
fn page_to_metadata(page: &Page) -> PageMetadata {
    match page {
        Page::DataPage { num_values, .. } => PageMetadata {
            num_rows: None,
            num_levels: Some(*num_values as usize),
            is_dict: false,
        },
        Page::DataPageV2 {
            num_values,
            num_rows,
            ..
        } => PageMetadata {
            num_rows: Some(*num_rows as usize),
            num_levels: Some(*num_values as usize),
            is_dict: false,
        },
        Page::DictionaryPage { .. } => PageMetadata {
            num_rows: None,
            num_levels: None,
            is_dict: true,
        },
    }
}

/// A single-element PageIterator.
struct SinglePageIterator {
    reader: Option<Result<Box<dyn PageReader>>>,
}

impl Iterator for SinglePageIterator {
    type Item = Result<Box<dyn PageReader>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.take()
    }
}

impl PageIterator for SinglePageIterator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::ArrowWriter;
    use crate::file::page_index::page_metadata::RowGroupPageIndex;
    use crate::file::properties::WriterProperties;
    use crate::file::reader::{FileReader, SerializedFileReader};
    use arrow_array::{ArrayRef, Int32Array};
    use bytes::Bytes;

    fn create_test_parquet() -> (Bytes, Arc<ParquetMetaData>) {
        let mut buf = vec![];
        let data: Vec<i32> = (0..100).collect();
        let array: ArrayRef = Arc::new(Int32Array::from(data));
        let batch = RecordBatch::try_from_iter(vec![("id", array)]).unwrap();

        let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let bytes = Bytes::from(buf);

        use crate::file::metadata::ParquetMetaDataReader;
        let metadata = ParquetMetaDataReader::new()
            .parse_and_finish(&bytes)
            .unwrap();

        (bytes, Arc::new(metadata))
    }

    fn create_multi_column_parquet() -> (Bytes, Arc<ParquetMetaData>) {
        let mut buf = vec![];
        let col1: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));
        let col2: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50]));
        let batch = RecordBatch::try_from_iter(vec![("a", col1), ("b", col2)]).unwrap();

        let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let bytes = Bytes::from(buf);

        use crate::file::metadata::ParquetMetaDataReader;
        let metadata = ParquetMetaDataReader::new()
            .parse_and_finish(&bytes)
            .unwrap();

        (bytes, Arc::new(metadata))
    }

    #[test]
    fn test_reader_creation() {
        let (bytes, metadata) = create_test_parquet();
        let page_index = RowGroupPageIndex::load(&bytes, metadata.clone(), 0).unwrap();
        let reader = RowGroupPageBasedReader::try_new(&page_index, 1024).unwrap();

        assert_eq!(reader.row_group_idx(), 0);
        assert_eq!(reader.rows_remaining(), 100);
        assert!(reader.has_more_rows());
    }

    #[test]
    fn test_no_pages_returns_error() {
        let (bytes, metadata) = create_test_parquet();
        let page_index = RowGroupPageIndex::load(&bytes, metadata.clone(), 0).unwrap();
        let mut reader = RowGroupPageBasedReader::try_new(&page_index, 1024).unwrap();

        let result = reader.read_record_batch();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No values available"));
    }

    #[test]
    fn test_partial_pages_returns_clear_error() {
        let (bytes, metadata) = create_multi_column_parquet();
        let page_index = RowGroupPageIndex::load(&bytes, metadata.clone(), 0).unwrap();

        // Get pages for column 0 only
        let file_reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let row_group_reader = file_reader.get_row_group(0).unwrap();
        let mut page_reader = row_group_reader.get_column_page_reader(0).unwrap();

        let mut col0_pages = vec![];
        while let Some(page) = page_reader.get_next_page().unwrap() {
            col0_pages.push(page);
        }

        let mut reader = RowGroupPageBasedReader::try_new(&page_index, 1024).unwrap();
        for (idx, page) in col0_pages.into_iter().enumerate() {
            reader.push_page(0, idx, page).unwrap();
        }
        // Don't push pages for column 1

        let result = reader.read_record_batch();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("No values available"),
            "Expected 'No values available' error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_out_of_order_pages() {
        // Create a file with multiple pages per column
        let mut buf = vec![];
        let col_a: ArrayRef = Arc::new(Int32Array::from((0..20).collect::<Vec<_>>()));
        let col_b: ArrayRef = Arc::new(Int32Array::from((100..120).collect::<Vec<_>>()));
        let batch = RecordBatch::try_from_iter(vec![("a", col_a), ("b", col_b)]).unwrap();

        let props = WriterProperties::builder()
            .set_data_page_row_count_limit(5) // 5 rows per page
            .build();

        let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let bytes = Bytes::from(buf);

        use crate::file::metadata::ParquetMetaDataReader;
        let metadata = Arc::new(
            ParquetMetaDataReader::new()
                .parse_and_finish(&bytes)
                .unwrap(),
        );

        let page_index = RowGroupPageIndex::load(&bytes, metadata.clone(), 0).unwrap();

        // Get pages from file
        let file_reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let row_group_reader = file_reader.get_row_group(0).unwrap();

        let mut col0_pages = vec![];
        let mut page_reader = row_group_reader.get_column_page_reader(0).unwrap();
        while let Some(page) = page_reader.get_next_page().unwrap() {
            col0_pages.push(page);
        }

        let mut col1_pages = vec![];
        let mut page_reader = row_group_reader.get_column_page_reader(1).unwrap();
        while let Some(page) = page_reader.get_next_page().unwrap() {
            col1_pages.push(page);
        }

        let mut reader = RowGroupPageBasedReader::try_new(&page_index, 1024).unwrap();

        // Push pages in REVERSE order
        for (idx, page) in col1_pages.into_iter().enumerate().rev() {
            reader.push_page(1, idx, page).unwrap();
        }
        for (idx, page) in col0_pages.into_iter().enumerate().rev() {
            reader.push_page(0, idx, page).unwrap();
        }

        // Read should succeed
        let batch = reader.read_record_batch().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 20);

        // Verify data
        let col_a = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let col_b = batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();

        for i in 0..20 {
            assert_eq!(col_a.value(i), i as i32);
            assert_eq!(col_b.value(i), 100 + i as i32);
        }

        assert!(!reader.has_more_rows());
    }

    #[test]
    fn test_insufficient_pages_then_push_more() {
        let mut buf = vec![];
        let col_a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        let col_b: ArrayRef =
            Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100]));
        let batch = RecordBatch::try_from_iter(vec![("a", col_a), ("b", col_b)]).unwrap();

        let props = WriterProperties::builder()
            .set_data_page_row_count_limit(5)
            .build();

        let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let bytes = Bytes::from(buf);

        use crate::file::metadata::ParquetMetaDataReader;
        let metadata = Arc::new(
            ParquetMetaDataReader::new()
                .parse_and_finish(&bytes)
                .unwrap(),
        );

        let page_index = RowGroupPageIndex::load(&bytes, metadata.clone(), 0).unwrap();

        let file_reader = SerializedFileReader::new(bytes.clone()).unwrap();
        let row_group_reader = file_reader.get_row_group(0).unwrap();

        let mut col0_pages = vec![];
        let mut page_reader = row_group_reader.get_column_page_reader(0).unwrap();
        while let Some(page) = page_reader.get_next_page().unwrap() {
            col0_pages.push(page);
        }

        let mut col1_pages = vec![];
        let mut page_reader = row_group_reader.get_column_page_reader(1).unwrap();
        while let Some(page) = page_reader.get_next_page().unwrap() {
            col1_pages.push(page);
        }

        let mut reader = RowGroupPageBasedReader::try_new(&page_index, 10).unwrap();

        // Push only first page of col0 and all pages of col1
        reader.push_page(0, 0, col0_pages.remove(0)).unwrap();
        for (idx, page) in col1_pages.into_iter().enumerate() {
            reader.push_page(1, idx, page).unwrap();
        }

        // Should fail - col0 doesn't have enough data
        let result = reader.read_record_batch();
        assert!(result.is_err());

        // Push remaining pages for col0
        for (idx, page) in col0_pages.into_iter().enumerate() {
            reader.push_page(0, idx + 1, page).unwrap();
        }

        // Now should succeed
        let batch = reader.read_record_batch().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 10);
    }

    #[test]
    fn test_page_index_out_of_bounds() {
        let (bytes, metadata) = create_test_parquet();
        let page_index = RowGroupPageIndex::load(&bytes, metadata.clone(), 0).unwrap();
        let mut reader = RowGroupPageBasedReader::try_new(&page_index, 1024).unwrap();

        let page = Page::DataPage {
            buf: bytes::Bytes::new(),
            num_values: 0,
            encoding: crate::basic::Encoding::PLAIN,
            def_level_encoding: crate::basic::Encoding::RLE,
            rep_level_encoding: crate::basic::Encoding::RLE,
            statistics: None,
        };

        // Column out of bounds
        let result = reader.push_page(999, 0, page.clone());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Column index"));

        // Page out of bounds
        let result = reader.push_page(0, 999, page);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Page index"));
    }
}
