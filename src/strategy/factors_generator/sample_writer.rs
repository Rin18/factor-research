use std::{error::Error, fs::File, path::Path, sync::Arc};

use arrow_schema::{FieldRef, Schema};
use parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties};
use serde_arrow::schema::{SchemaLike, TracingOptions};

use super::Sample;

const SAMPLE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/samples/samples.parquet");
const ROW_GROUP_SIZE: usize = 65536;

pub(super) struct SampleWriter {
    writer: Option<ArrowWriter<File>>,
    fields: Vec<FieldRef>,
    /// Samples not yet serialized; written as one record batch once a row group is full.
    buffer: Vec<Sample>,
}

impl SampleWriter {
    pub(super) fn new() -> Self {
        Self::with_path(Path::new(SAMPLE_PATH))
    }

    fn with_path(path: &Path) -> Self {
        let fields = Vec::<FieldRef>::from_type::<Sample>(TracingOptions::default())
            .expect("failed to derive sample schema");
        let properties = WriterProperties::builder()
            .set_max_row_group_size(ROW_GROUP_SIZE)
            .set_compression(Compression::ZSTD(Default::default()))
            .build();
        std::fs::create_dir_all(path.parent().expect("sample path must have a parent"))
            .expect("failed to create sample directory");
        let file = File::create(path).expect("failed to create sample parquet file");
        let writer = ArrowWriter::try_new(
            file,
            Arc::new(Schema::new(fields.clone())),
            Some(properties),
        )
        .expect("failed to create sample parquet writer");
        Self {
            writer: Some(writer),
            fields,
            buffer: Vec::with_capacity(ROW_GROUP_SIZE),
        }
    }

    pub(super) fn add_sample(&mut self, sample: Sample) {
        assert!(self.writer.is_some(), "sample writer already finished");
        self.buffer.push(sample);
        if self.buffer.len() == ROW_GROUP_SIZE {
            self.flush();
        }
    }

    pub(super) fn finish(&mut self) {
        self.flush();
        if let Some(writer) = self.writer.take() {
            writer
                .close()
                .expect("failed to finish sample parquet file");
        }
    }

    fn flush(&mut self) {
        self.write_buffer().expect("failed to write samples");
    }

    /// Writes the buffered samples as one record batch.
    fn write_buffer(&mut self) -> Result<(), Box<dyn Error>> {
        let Some(writer) = self.writer.as_mut() else {
            return Ok(());
        };
        if self.buffer.is_empty() {
            return Ok(());
        }
        let batch = serde_arrow::to_record_batch(&self.fields, &self.buffer)?;
        writer.write(&batch)?;
        self.buffer.clear();
        Ok(())
    }
}

impl Drop for SampleWriter {
    fn drop(&mut self) {
        if let Err(error) = self.write_buffer() {
            eprintln!("failed to write samples: {error}");
        }
        if let Some(writer) = self.writer.take() {
            if let Err(error) = writer.close() {
                eprintln!("failed to finish sample parquet file: {error}");
            }
        }
    }
}
