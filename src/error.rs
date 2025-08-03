use crate::graph::GraphError;
use ragit_fs::FileError;

#[derive(Debug)]
pub enum Error {
    GraphError(GraphError),
    FileError(FileError),
    StdIoError(std::io::Error),
    CliError {
        message: String,
        span: Option<ragit_cli::RenderedSpan>,
    },
    JsonSerdeError(serde_json::Error),
    BincodeError(Box<bincode::ErrorKind>),
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

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::StdIoError(e)
    }
}

impl From<ragit_cli::Error> for Error {
    fn from(e: ragit_cli::Error) -> Self {
        Error::CliError {
            message: e.kind.render(),
            span: e.span,
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Error {
        Error::JsonSerdeError(e)
    }
}

impl From<Box<bincode::ErrorKind>> for Error {
    fn from(e: Box<bincode::ErrorKind>) -> Error {
        Error::BincodeError(e)
    }
}
