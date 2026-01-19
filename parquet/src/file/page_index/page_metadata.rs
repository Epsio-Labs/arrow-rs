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

//! Page metadata for two-stage Parquet loading.
//!
//! This module provides [`PageInfo`] and [`RowGroupPageIndex`] for loading page metadata
//! without decompressing page data, enabling external parallel decompression.

use bytes::Bytes;
use std::io::Read;
use std::ops::Range;
use std::sync::Arc;

use crate::basic::{Compression, Encoding, PageType};
use crate::errors::{ParquetError, Result};
use crate::file::metadata::thrift::PageHeader;
use crate::file::metadata::ParquetMetaData;
use crate::file::reader::ChunkReader;
use crate::parquet_thrift::ThriftReadInputProtocol;

/// Metadata about a single page in a Parquet file.
///
/// This contains all information needed to locate and decompress a page,
/// without actually reading the page data.
#[derive(Debug, Clone)]
pub struct PageInfo {
    /// Row group index this page belongs to
    pub row_group_idx: usize,
    /// Column (leaf) index within the schema
    pub column_idx: usize,
    /// Page index within this column chunk (0-indexed)
    pub page_idx: usize,
    /// Absolute offset of the page in the file (start of header)
    pub offset: u64,
    /// Size of the page header in bytes
    pub header_size: u32,
    /// Compressed size of page data (not including header)
    pub compressed_size: u32,
    /// Uncompressed size of page data (not including header)
    pub uncompressed_size: u32,
    /// Number of values in this page
    pub num_values: u32,
    /// Whether this is a dictionary page
    pub is_dictionary: bool,
    /// Page type
    pub page_type: PageType,
    /// Encoding used for this page
    pub encoding: Encoding,
    /// Compression codec (needed for decompression)
    pub compression: Compression,
    /// For DATA_PAGE: definition level encoding
    pub def_level_encoding: Option<Encoding>,
    /// For DATA_PAGE: repetition level encoding
    pub rep_level_encoding: Option<Encoding>,
    /// For DATA_PAGE_V2: definition levels byte length
    pub def_levels_byte_len: Option<u32>,
    /// For DATA_PAGE_V2: repetition levels byte length
    pub rep_levels_byte_len: Option<u32>,
    /// For DATA_PAGE_V2: number of nulls
    pub num_nulls: Option<u32>,
    /// For DATA_PAGE_V2: number of rows
    pub num_rows: Option<u32>,
    /// For DATA_PAGE_V2: whether the page is compressed
    pub is_compressed: Option<bool>,
    /// For DICTIONARY_PAGE: whether sorted
    pub is_sorted: Option<bool>,
}

impl PageInfo {
    /// Returns the byte range of the page data (excluding header) in the file.
    pub fn data_byte_range(&self) -> Range<u64> {
        let start = self.offset + self.header_size as u64;
        let end = start + self.compressed_size as u64;
        start..end
    }

    /// Returns the byte range of the entire page (including header) in the file.
    pub fn full_byte_range(&self) -> Range<u64> {
        let start = self.offset;
        let end = start + self.header_size as u64 + self.compressed_size as u64;
        start..end
    }

    /// Returns the byte range of the page data relative to a given base offset.
    ///
    /// This is useful when working with row group bytes where offsets are relative
    /// to the start of the row group, not the file.
    pub fn data_byte_range_relative(&self, base_offset: u64) -> Range<usize> {
        let start = (self.offset - base_offset + self.header_size as u64) as usize;
        let end = start + self.compressed_size as usize;
        start..end
    }

    /// Returns the byte range of the entire page (including header) relative to a base offset.
    pub fn full_byte_range_relative(&self, base_offset: u64) -> Range<usize> {
        let start = (self.offset - base_offset) as usize;
        let end = start + self.header_size as usize + self.compressed_size as usize;
        start..end
    }
}

/// A compressed page with its metadata and raw compressed data.
///
/// This is used for zero-copy extraction of pages from row group bytes,
/// enabling parallel decompression.
#[derive(Debug, Clone)]
pub struct CompressedPage {
    /// Page metadata
    pub info: PageInfo,
    /// Compressed page data (zero-copy slice from row group bytes)
    pub data: Bytes,
}

impl CompressedPage {
    /// Get the column index this page belongs to.
    pub fn column_idx(&self) -> usize {
        self.info.column_idx
    }

    /// Get the page index within the column (for ordering).
    /// Note: This is the index in the pages list, including dictionary pages.
    pub fn page_list_idx(&self) -> usize {
        self.info.page_idx
    }

    /// Whether this is a dictionary page.
    pub fn is_dictionary(&self) -> bool {
        self.info.is_dictionary
    }

    /// Get the compression codec.
    pub fn compression(&self) -> Compression {
        self.info.compression
    }

    /// Get the uncompressed size (for allocating decompression buffer).
    pub fn uncompressed_size(&self) -> usize {
        self.info.uncompressed_size as usize
    }
}

/// Page metadata for a single row group in a Parquet file.
///
/// This is the result of Stage 1 loading - it contains metadata about all pages
/// in a specific row group without the actual page data.
#[derive(Debug, Clone)]
pub struct RowGroupPageIndex {
    /// Row group index this page index is for
    row_group_idx: usize,
    /// Pages organized by [column_idx][page_list_idx]
    pages: Vec<Vec<PageInfo>>,
    /// Reference to file metadata
    metadata: Arc<ParquetMetaData>,
    /// Start offset of the row group in the file (used for relative offset calculations)
    row_group_offset: u64,
}

impl RowGroupPageIndex {
    /// Load page metadata for a specific row group by scanning page headers.
    ///
    /// This reads all page headers to build a complete index of pages.
    /// The actual page data is not read or decompressed.
    pub fn load<R: ChunkReader>(
        reader: &R,
        metadata: Arc<ParquetMetaData>,
        row_group_idx: usize,
    ) -> Result<Self> {
        let num_row_groups = metadata.num_row_groups();
        if row_group_idx >= num_row_groups {
            return Err(ParquetError::General(format!(
                "Row group index {} out of bounds ({})",
                row_group_idx, num_row_groups
            )));
        }

        let row_group = metadata.row_group(row_group_idx);
        let num_columns = row_group.num_columns();
        let mut pages = Vec::with_capacity(num_columns);

        // Compute row group start offset (minimum of all column chunk offsets)
        let row_group_offset = (0..num_columns)
            .map(|i| row_group.column(i).byte_range().0)
            .min()
            .unwrap_or(0);

        for column_idx in 0..num_columns {
            let column = row_group.column(column_idx);
            let compression = column.compression();
            let (chunk_start, chunk_len) = column.byte_range();
            let chunk_end = chunk_start + chunk_len;

            let column_pages =
                Self::scan_column_pages(reader, row_group_idx, column_idx, compression, chunk_start, chunk_end)?;
            pages.push(column_pages);
        }

        Ok(Self {
            row_group_idx,
            pages,
            metadata,
            row_group_offset,
        })
    }

    /// Scan page headers for a single column chunk.
    fn scan_column_pages<R: ChunkReader>(
        reader: &R,
        row_group_idx: usize,
        column_idx: usize,
        compression: Compression,
        chunk_start: u64,
        chunk_end: u64,
    ) -> Result<Vec<PageInfo>> {
        let mut pages = Vec::new();
        let mut offset = chunk_start;
        let mut page_idx = 0;

        while offset < chunk_end {
            let remaining = chunk_end - offset;
            // Read enough bytes for the header (headers are typically small, but can vary)
            // We read a reasonable chunk and parse from it
            let read_len = std::cmp::min(remaining, 1024) as usize;
            let buffer = reader.get_bytes(offset, read_len)?;

            let (header_len, header) = Self::read_page_header(&buffer)?;

            let page_info = Self::header_to_page_info(
                &header,
                row_group_idx,
                column_idx,
                page_idx,
                offset,
                header_len as u32,
                compression,
            )?;

            let is_dict = page_info.is_dictionary;
            pages.push(page_info);

            // Move to next page
            offset += header_len as u64 + header.compressed_page_size as u64;
            if !is_dict {
                page_idx += 1;
            }
        }

        Ok(pages)
    }

    /// Read a page header from a buffer.
    fn read_page_header(buffer: &[u8]) -> Result<(usize, PageHeader)> {
        struct TrackedRead<'a> {
            data: &'a [u8],
            pos: usize,
        }

        impl Read for TrackedRead<'_> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let available = self.data.len() - self.pos;
                let to_read = std::cmp::min(available, buf.len());
                buf[..to_read].copy_from_slice(&self.data[self.pos..self.pos + to_read]);
                self.pos += to_read;
                Ok(to_read)
            }
        }

        let mut tracked = TrackedRead {
            data: buffer,
            pos: 0,
        };
        let mut prot = ThriftReadInputProtocol::new(&mut tracked);
        let header = PageHeader::read_thrift_without_stats(&mut prot)?;
        Ok((tracked.pos, header))
    }

    /// Convert a PageHeader to PageInfo.
    fn header_to_page_info(
        header: &PageHeader,
        row_group_idx: usize,
        column_idx: usize,
        page_idx: usize,
        offset: u64,
        header_size: u32,
        compression: Compression,
    ) -> Result<PageInfo> {
        let page_type = header.r#type;
        let is_dictionary = page_type == PageType::DICTIONARY_PAGE;

        let (
            encoding,
            num_values,
            def_level_encoding,
            rep_level_encoding,
            def_levels_byte_len,
            rep_levels_byte_len,
            num_nulls,
            num_rows,
            is_compressed,
            is_sorted,
        ) = match page_type {
            PageType::DATA_PAGE => {
                let h = header.data_page_header.as_ref().ok_or_else(|| {
                    ParquetError::General("DATA_PAGE missing data_page_header".to_string())
                })?;
                (
                    h.encoding,
                    h.num_values as u32,
                    Some(h.definition_level_encoding),
                    Some(h.repetition_level_encoding),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            }
            PageType::DATA_PAGE_V2 => {
                let h = header.data_page_header_v2.as_ref().ok_or_else(|| {
                    ParquetError::General("DATA_PAGE_V2 missing data_page_header_v2".to_string())
                })?;
                (
                    h.encoding,
                    h.num_values as u32,
                    None,
                    None,
                    Some(h.definition_levels_byte_length as u32),
                    Some(h.repetition_levels_byte_length as u32),
                    Some(h.num_nulls as u32),
                    Some(h.num_rows as u32),
                    h.is_compressed,
                    None,
                )
            }
            PageType::DICTIONARY_PAGE => {
                let h = header.dictionary_page_header.as_ref().ok_or_else(|| {
                    ParquetError::General(
                        "DICTIONARY_PAGE missing dictionary_page_header".to_string(),
                    )
                })?;
                (
                    h.encoding,
                    h.num_values as u32,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    h.is_sorted,
                )
            }
            other => {
                return Err(ParquetError::General(format!(
                    "Unsupported page type: {:?}",
                    other
                )));
            }
        };

        Ok(PageInfo {
            row_group_idx,
            column_idx,
            page_idx,
            offset,
            header_size,
            compressed_size: header.compressed_page_size as u32,
            uncompressed_size: header.uncompressed_page_size as u32,
            num_values,
            is_dictionary,
            page_type,
            encoding,
            compression,
            def_level_encoding,
            rep_level_encoding,
            def_levels_byte_len,
            rep_levels_byte_len,
            num_nulls,
            num_rows,
            is_compressed,
            is_sorted,
        })
    }

    /// Get the row group index this page index is for.
    pub fn row_group_idx(&self) -> usize {
        self.row_group_idx
    }

    /// Get all pages organized by column.
    ///
    /// Returns a slice of columns, where each column contains its pages.
    pub fn pages(&self) -> &[Vec<PageInfo>] {
        &self.pages
    }

    /// Get pages for a specific column.
    pub fn pages_for_column(&self, column_idx: usize) -> &[PageInfo] {
        &self.pages[column_idx]
    }

    /// Get the number of columns.
    pub fn num_columns(&self) -> usize {
        self.pages.len()
    }

    /// Get the total number of rows in this row group.
    pub fn num_rows(&self) -> usize {
        self.metadata.row_group(self.row_group_idx).num_rows() as usize
    }

    /// Get underlying Parquet metadata.
    pub fn metadata(&self) -> &Arc<ParquetMetaData> {
        &self.metadata
    }

    /// Get the starting offset of the row group in the file.
    ///
    /// This is the minimum offset across all column chunks in the row group.
    pub fn row_group_offset(&self) -> u64 {
        self.row_group_offset
    }

    /// Extract all compressed pages from row group bytes as zero-copy slices.
    ///
    /// The `row_group_bytes` should contain the raw bytes of the row group,
    /// starting from `row_group_offset()`. This method returns compressed pages
    /// organized by column, with each page containing a zero-copy slice of the
    /// compressed data.
    ///
    /// # Arguments
    /// * `row_group_bytes` - Bytes containing the row group data, starting at `row_group_offset()`
    ///
    /// # Returns
    /// A vector of vectors, where `result[column_idx]` contains the compressed pages
    /// for that column in order (dictionary page first if present, then data pages).
    pub fn extract_compressed_pages(&self, row_group_bytes: &Bytes) -> Vec<Vec<CompressedPage>> {
        self.pages
            .iter()
            .map(|column_pages| {
                column_pages
                    .iter()
                    .map(|page_info| {
                        let range = page_info.data_byte_range_relative(self.row_group_offset);
                        CompressedPage {
                            info: page_info.clone(),
                            data: row_group_bytes.slice(range),
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Extract a single compressed page from row group bytes.
    ///
    /// # Arguments
    /// * `column_idx` - Column index
    /// * `page_idx` - Page index within the column (0 = first page, which may be dictionary)
    /// * `row_group_bytes` - Bytes containing the row group data, starting at `row_group_offset()`
    ///
    /// # Returns
    /// The compressed page with zero-copy data slice, or None if indices are out of bounds.
    pub fn extract_compressed_page(
        &self,
        column_idx: usize,
        page_idx: usize,
        row_group_bytes: &Bytes,
    ) -> Option<CompressedPage> {
        let page_info = self.pages.get(column_idx)?.get(page_idx)?;
        let range = page_info.data_byte_range_relative(self.row_group_offset);
        Some(CompressedPage {
            info: page_info.clone(),
            data: row_group_bytes.slice(range),
        })
    }

    /// Get the total number of pages across all columns.
    pub fn total_pages(&self) -> usize {
        self.pages.iter().map(|c| c.len()).sum()
    }

    /// Get page count for each column.
    ///
    /// Returns a vector where `result[column_idx]` is the number of pages for that column.
    pub fn pages_per_column(&self) -> Vec<usize> {
        self.pages.iter().map(|c| c.len()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_info_byte_ranges() {
        let page = PageInfo {
            row_group_idx: 0,
            column_idx: 0,
            page_idx: 0,
            offset: 100,
            header_size: 20,
            compressed_size: 500,
            uncompressed_size: 1000,
            num_values: 100,
            is_dictionary: false,
            page_type: PageType::DATA_PAGE,
            encoding: Encoding::PLAIN,
            compression: Compression::SNAPPY,
            def_level_encoding: Some(Encoding::RLE),
            rep_level_encoding: Some(Encoding::RLE),
            def_levels_byte_len: None,
            rep_levels_byte_len: None,
            num_nulls: None,
            num_rows: None,
            is_compressed: None,
            is_sorted: None,
        };

        assert_eq!(page.data_byte_range(), 120..620);
        assert_eq!(page.full_byte_range(), 100..620);
    }
}
