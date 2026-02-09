use voiceflow_core::prosody::fix_tokenization_artifacts;

fn main() {
    let inputs = [
        "Pride after satisfaction uplifted hiMLike long, slow waves.",
        "The Europe they had come froMLay out there beyond the Irish Sea.",
        "A great saint, St. Francis Xavier.",
        "grea.tsaint cas.tsin",
    ];

    for input in inputs {
        let output = fix_tokenization_artifacts(input);
        println!("Input:  {}", input);
        println!("Output: {}", output);
        println!();
    }
}
