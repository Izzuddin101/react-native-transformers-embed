import "text-encoding-polyfill";
import { Tensor } from "onnxruntime-react-native";
import { Base } from "./base";

/**
 * Class to handle text embedding model on top of onnxruntime
 */
export class TextEmbedding extends Base {
  /**
   * Generate embeddings from input tokens
   *
   * @param tokens Input tokens to generate embeddings from
   * @returns Float32Array containing the embedding vector
   */
  public async embed(tokens: bigint[]): Promise<Float32Array> {
    if (!this.sess) {
      throw new Error("Session is undefined");
    }

    const feed = this.feed;
    
    // Ensure tokens array is not empty to prevent dimension issues
    if (tokens.length === 0) {
      throw new Error("Input tokens array cannot be empty");
    }

    const inputIdsTensor = new Tensor(
      "int64",
      BigInt64Array.from(tokens.map(BigInt)),
      [1, tokens.length],
    );
    feed.input_ids = inputIdsTensor;

    // Create attention mask (1 for all tokens)
    const attentionMask = BigInt64Array.from({ length: tokens.length }, () => 1n);
    feed.attention_mask = new Tensor(
      "int64",
      attentionMask,
      [1, tokens.length],
    );

    // Run inference to get embeddings
    const outputs = await this.sess.run(feed);

    // The model typically outputs the embeddings as 'last_hidden_state' or 'embeddings'
    // We take the mean of the token embeddings to get a single vector
    const embeddings = outputs.last_hidden_state || outputs.embeddings;

    if (!embeddings) {
      throw new Error("No embedding output found in model outputs");
    }

    // Calculate mean across token dimension (dim 1) to get a single embedding vector
    const data = embeddings.data as Float32Array;
    const [, seqLen, hiddenSize] = embeddings.dims;
    
    // Check if dimensions are defined
    if (seqLen === undefined || hiddenSize === undefined) {
      throw new Error(`Invalid embedding dimensions: [${embeddings.dims.join(', ')}]`);
    }
    
    const result = new Float32Array(hiddenSize);

    for (let h = 0; h < hiddenSize; h++) {
      let sum = 0;
      for (let s = 0; s < seqLen; s++) {
        sum += data[s * hiddenSize + h];
      }
      result[h] = sum / seqLen;
    }

    return result;
  }
}
