use std::env::args;

use serdeio::write_record_to_file;
use silva::parser::read_xgboost_model;

fn main() {
    let xgboost_model_path = args()
        .nth(1)
        .expect("Please provide the path to the xgboost model file");
    let output_path = args()
        .nth(2)
        .expect("Please provide the output path for the converted model");

    // read xgboost model file
    let model = read_xgboost_model(xgboost_model_path).expect("Failed to read xgboost model");

    // write loaded model to output path
    write_record_to_file(output_path, &model).expect("Failed to write converted model");
}
