use crate::{DataType, Field, Fields};

/// We represent the JSON type as a custom arrow struct, with one field containing the
/// JSON as text
pub fn json_type() -> DataType {
    let json_struct = Fields::from(vec![Field::new("json", DataType::Utf8, false)]);
    DataType::Struct(json_struct)
}

/// We represent the tsvector type as a custom arrow struct, with one field containing the
/// tsvector as text
pub fn tsvector_type() -> DataType {
    let tsvector_struct = Fields::from(vec![Field::new("tsvector", DataType::Utf8, false)]);
    DataType::Struct(tsvector_struct)
}


/// We represent the bit / bit varying type as a custom arrow struct, with one field containing the
/// bit string as text
pub fn bit_type() -> DataType {
    let bit_struct = Fields::from(vec![Field::new("bit", DataType::Utf8, false)]);
    DataType::Struct(bit_struct)
}

pub fn enum_type(schema_name: String, enum_name: String) -> DataType {
    let enum_struct = Fields::from(vec![
        Field::new("enum", DataType::Utf8, false),
        Field::new(schema_name, DataType::Utf8, false),
        Field::new(enum_name, DataType::Utf8, false),
    ]);
    DataType::Struct(enum_struct)
}

pub fn extract_enum_schema_and_name(t: &DataType) -> Option<(String, String)> {
    if let DataType::Struct(fields) = t {
        if fields.len() == 3 && fields[0].name() == "enum" {
            return Some((fields[1].name().to_string(), fields[2].name().to_string()));
        }
    }
    None
}
