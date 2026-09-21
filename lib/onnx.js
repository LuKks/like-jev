'use strict'

const fsp = require('fs/promises')
const os = require('os')
const path = require('path')
const ort = require('onnxruntime-node')
const { ensureBundle, cacheDir } = require('./bundle.js')
const { loadTokenizer } = require('./questions.js')

const BUNDLE_FILES = [
  'laya.onnx',
  'laya.onnx.data',
  'laya_config.json',
  'tokenizer/tokenizer.json',
  'tokenizer/tokenizer_config.json'
]
const MODEL_BASE = 'https://huggingface.co/receptron/laya-onnx/resolve/main'
const CACHE_DIR = path.join(cacheDir('laya'), 'onnx')
const LEGACY_DIR = path.join(os.homedir(), '.cache', 'receptron-laya')

exports.load = async function (opts = {}) {
  ort.env.logLevel = opts.logLevel || 'error'

  const modelDir = await resolveModelDir(opts)
  const config = JSON.parse(await fsp.readFile(path.join(modelDir, 'laya_config.json'), 'utf8'))
  const { tokenizer, ids } = await loadTokenizer(modelDir)
  const session = await createSession(opts, modelDir)

  return {
    name: 'laya',
    padToMultiple: null,
    tokenizer,
    ids,
    config,
    modelDir,

    async forward (batch) {
      const output = await session.run(buildInputs(batch))
      return readOutputs(output, batch)
    },

    async close () {
      await session.release()
    }
  }
}

async function resolveModelDir (opts) {
  if (opts.modelDir) return path.resolve(opts.modelDir)

  return ensureBundle({
    files: BUNDLE_FILES,
    base: MODEL_BASE,
    cacheDir: CACHE_DIR,
    fallbackDir: LEGACY_DIR
  })
}

async function createSession (opts, modelDir) {
  return ort.InferenceSession.create(path.join(modelDir, 'laya.onnx'), {
    executionProviders: opts.executionProviders || ['webgpu', 'cpu'],
    graphOptimizationLevel: 'all',
    ...opts.sessionOptions
  })
}

function buildInputs (batch) {
  return {
    input_ids: new ort.Tensor('int64', toBigInt64(batch.inputIds), [batch.size, batch.length]),
    attention_mask: new ort.Tensor('int64', toBigInt64(batch.attention), [batch.size, batch.length]),
    marker_pos: new ort.Tensor('int64', toBigInt64(batch.markerPos), [batch.size, batch.options]),
    marker_mask: new ort.Tensor('bool', toUint8(batch.markerMask), [batch.size, batch.options]),
    qtype: new ort.Tensor('int64', toBigInt64(batch.qtype), [batch.size])
  }
}

function readOutputs (output, batch) {
  const act = output.act_probs.data
  const actWidth = output.act_probs.dims[1] || 1

  return {
    logits: sliceRows(output.logits.data, batch.size, batch.options),
    actProbabilities: sliceRows(act, batch.size, actWidth)
  }
}

function sliceRows (flat, size, width) {
  const rows = []

  for (let row = 0; row < size; row++) {
    rows.push(Array.from(flat.subarray(row * width, (row + 1) * width)))
  }

  return rows
}

function toBigInt64 (values) {
  const flat = Array.isArray(values) ? values.flat() : Array.from(values)
  return BigInt64Array.from(flat, value => BigInt(value))
}

function toUint8 (nested) {
  return Uint8Array.from(nested.flat())
}
