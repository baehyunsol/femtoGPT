use crate::graph::GraphError;
use ragit_fs::FileError;

#[derive(Debug)]
pub enum Error {
    GraphError(GraphError),
    FileError(FileError),
    CliError {
        message: String,
        span: (String, usize, usize),  // (args, error_from, error_to)
    },
}

impl From<GraphError> for Error {
    fn from(e: GraphError) -> Error {
        Error::GraphError(e)
    }
}

impl From<FileError> for Error {
    fn from(e: FileError) -> Error {
        Error::FileError(e)
    }
}

impl From<ragit_cli::Error> for Error {
    fn from(e: ragit_cli::Error) -> Self {
        Error::CliError {
            message: e.kind.render(),
            span: e.span.unwrap_rendered(),
        }
    }
}
