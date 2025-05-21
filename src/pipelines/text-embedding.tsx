import { env, AutoTokenizer, PreTrainedTokenizer } from "@xenova/transformers";
import { TextEmbedding as Model } from "../models/text-embedding";
import { LoadOptions } from "../models/base";

/** Initialization Options for Text Embedding */
export interface TextEmbeddingOptions extends LoadOptions {
  /** Shows special tokens in the output. */
  show_special: boolean;
}

// Set up environment for transformers.js tokenizer
env.allowRemoteModels = true;
env.allowLocalModels = false;

// Declare tokenizer and model
let tokenizer: PreTrainedTokenizer;
const model: Model = new Model();

// Initialize options with default values
let _options: TextEmbeddingOptions = {
  show_special: false,
  max_tokens: 512, // typical max length for embedding models
  fetch: async (url) => url,
  verbose: false,
  externalData: false,
  executionProviders: ["cpu"],
};

/**
 * Generates embeddings from the input text.
 *
 * @param text - The input text to generate embeddings from.
 * @returns Float32Array containing the embedding vector.
 */
async function embed(text: string): Promise<Float32Array> {
  if (!tokenizer) {
    throw new Error("Tokenizer undefined, please initialize first.");
  }

  try {
    // Tokenize the input text
    const tokenized = await tokenizer(text, {
      return_tensor: false,
      padding: true,
      truncation: true,
      max_length: _options.max_tokens,
    });
    
    // Check if input_ids exists and is not empty
    if (!tokenized.input_ids || tokenized.input_ids.length === 0) {
      throw new Error("Tokenization resulted in empty input_ids");
    }
    
    // Generate embeddings from the tokenized input
    return await model.embed(tokenized.input_ids);
  } catch (error) {
    console.error("Error during embedding:", error);
    throw error;
  }
}

/**
 * Loads the model and tokenizer with the specified options.
 *
 * @param model_name - The name of the model to load.
 * @param onnx_path - The path to the ONNX model.
 * @param options - Optional initialization options.
 */
async function init(
  model_name: string,
  onnx_path: string,
  options?: Partial<TextEmbeddingOptions>,
): Promise<void> {
  try {
    _options = { ..._options, ...options };
    
    // Load tokenizer first
    console.log(`Loading tokenizer from ${model_name}...`);
    tokenizer = await AutoTokenizer.from_pretrained(model_name);
    
    // Then load the model
    console.log(`Loading model from ${model_name}, path: ${onnx_path}...`);
    await model.load(model_name, onnx_path, _options);
    
    console.log("Initialization completed successfully");
  } catch (error) {
    console.error("Error initializing pipeline:", error);
    throw error;
  }
}

/**
 * Releases the resources used by the model.
 */
async function release(): Promise<void> {
  await model.release();
  // Clear the tokenizer reference
  tokenizer = undefined as any;
}

// Export functions for external use
export default {
  init,
  embed,
  release,
};
