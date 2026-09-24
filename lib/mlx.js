const fs = require('fs')
const fsp = require('fs/promises')
const os = require('os')
const path = require('path')
const { makeBatch } = require('./batch.js')
const { bundleComplete, cacheDir, downloadFile, ensureBundle } = require('./bundle.js')
const { loadTokenizer } = require('./questions.js')

let mx = null

const BUNDLE_FILES = [
  'model.safetensors',
  'encoder/config.json',
  'rl_agent_config.json',
  'tokenizer/tokenizer.json',
  'tokenizer/tokenizer_config.json'
]
const MODEL_BASE = 'https://huggingface.co/aac6fef/laya-mlx/resolve/main'
const CACHE_DIR = path.join(cacheDir('laya'), 'mlx')
const LEGACY_DIR = path.join(os.homedir(), '.cache', 'laya-mlx')
const DTYPES = ['float16', 'float32']
const PKG_DIR = path.dirname(path.dirname(require.resolve('@frost-beta/mlx')))
const NATIVE_DIR = path.join(PKG_DIR, 'build', 'Release')

exports.hasModel = function () {
  return bundleComplete(BUNDLE_FILES, CACHE_DIR) || bundleComplete(BUNDLE_FILES, LEGACY_DIR)
}

exports.load = async function (opts = {}) {
  await ensureNative()
  mx = require('@frost-beta/mlx').core

  const modelDir = await resolveModelDir(opts)
  const agent = await readJson(modelDir, 'rl_agent_config.json')
  const enc = await readJson(modelDir, 'encoder/config.json')
  if (enc.model_type !== 'modernbert') throw new Error(`Unsupported encoder: ${enc.model_type}`)

  const { tokenizer, ids } = await loadTokenizer(modelDir)
  const weights = loadWeights(modelDir, resolveDtype(opts.dtype))
  const compute = buildCompute(weights, encoderPlan(enc), opts)

  const config = {
    max_len: agent.max_len,
    head_max_len: agent.head_max_len,
    temperature: agent.temperature,
    temperature_by_options: agent.temperature_by_options
  }

  const engine = {
    name: 'laya-mlx',
    padToMultiple: 16,
    tokenizer,
    ids,
    config,
    modelDir,

    async forward (batch) {
      const inputIds = mx.array(new Int32Array(batch.inputIds), mx.int32).reshape(batch.size, batch.length)
      const attention = mx.array(batch.attention, mx.bool_).reshape(batch.size, batch.length)
      const markerPos = mx.array(new Int32Array(batch.markerPos), mx.int32).reshape(batch.size, batch.options)
      const markerMask = mx.array(batch.markerMask, mx.bool_).reshape(batch.size, batch.options)
      const qtype = mx.array(new Int32Array(batch.qtype), mx.int32)

      const { logitsOut, actOut } = mx.tidy(() => compute(inputIds, attention, markerPos, markerMask, qtype))

      mx.dispose(inputIds, attention, markerPos, markerMask, qtype)
      await mx.asyncEval(logitsOut)
      await mx.asyncEval(actOut)

      const logits = logitsOut.tolist()
      const act = actOut.tolist()
      mx.dispose(logitsOut, actOut)

      return { logits, actProbabilities: act.map(row => [actProbability(row)]) }
    },

    async close () {}
  }

  await warmUp(engine)

  return engine
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

async function readJson (modelDir, file) {
  const raw = await fsp.readFile(path.join(modelDir, file), 'utf8')
  return JSON.parse(raw)
}

function resolveDtype (value) {
  const dtype = { float16: mx.float16, float32: mx.float32 }[value || 'float16']
  if (!dtype) throw new Error(`dtype must be one of ${DTYPES}`)
  return dtype
}

function loadWeights (modelDir, dtype) {
  const raw = mx.load(path.join(modelDir, 'model.safetensors'))
  const weights = {}

  for (const [name, value] of Object.entries(raw)) {
    weights[name] = value.dtype === dtype ? value : value.astype(dtype)
  }

  mx.eval(...Object.values(weights))
  return weights
}

function buildCompute (weights, plan, opts) {
  let compute = (a, b, c, d, e) => computeGraph(weights, plan, a, b, c, d, e)
  if (opts.compile !== false) compute = mx.compile(compute)
  return compute
}

async function warmUp (engine) {
  const ids = engine.ids
  const tiny = makeBatch([{
    ids: [ids.cls, 5, 6, ids.sep, ids.mask, 7, ids.sep],
    markers: [4],
    qtype: 0
  }], ids.pad, engine.config.max_len, engine.padToMultiple)

  await engine.forward(tiny)
}

function computeGraph (w, enc, inputIds, attention, markerPos, markerMask, qtype) {
  const shape = {
    b: attention.shape[0],
    len: attention.shape[1],
    dims: enc.hiddenSize
  }
  const count = markerMask.shape[1]

  const fullMask = attention.reshape(shape.b, 1, 1, shape.len)
  const localMask = buildLocalMask(attention, enc.localAttention)

  let h = embedTokens(w, enc, inputIds)
  for (let i = 0; i < enc.layers.length; i++) {
    h = applyEncoderLayer(w, enc, i, h, shape, fullMask, localMask)
  }

  h = mx.fast.layerNorm(h, w['encoder.final_norm.weight'], null, enc.normEps)
  h = mx.add(h, mx.take(w['type_emb.weight'], qtype, 0).index(mx.Slice(), null))

  for (let i = 0; i < enc.headLayers; i++) {
    h = applyHeadLayer(w, enc, i, h, shape, fullMask)
  }

  const flat = gatherMarkers(h, markerPos, shape, count)
  const logits = scoreMarkers(w, flat, markerMask)
  const features = agentFeatures(logits, markerMask)
  const action = actHead(w, h, features)

  return { logitsOut: logits.astype(mx.float32), actOut: action.astype(mx.float32) }
}

function embedTokens (w, enc, inputIds) {
  return mx.fast.layerNorm(
    mx.take(w['encoder.embeddings.tok_embeddings.weight'], inputIds, 0),
    w['encoder.embeddings.norm.weight'],
    null,
    enc.normEps
  )
}

function buildLocalMask (attention, localAttention) {
  const b = attention.shape[0]
  const len = attention.shape[1]

  const positions = mx.arange(len)
  const distance = mx.abs(mx.subtract(mx.expandDims(positions, 1), mx.expandDims(positions, 0)))
  const near = mx.lessEqual(distance, Math.floor(localAttention / 2)).index(null, null)
  const attended = mx.logicalOr(near, mx.logicalNot(attention).index(mx.Slice(), mx.Slice(), null))

  return mx.logicalAnd(attended, attention.index(mx.Slice(), null)).reshape(b, 1, len, len)
}

function applyEncoderLayer (w, enc, index, h, shape, fullMask, localMask) {
  const { b, len, dims } = shape
  const layer = enc.layers[index]
  const prefix = `encoder.layers.${index}.`

  const normed = layer.index === 0
    ? h
    : mx.fast.layerNorm(h, w[prefix + 'attn_norm.weight'], null, enc.normEps)

  const qkv = mx.matmul(normed, w[prefix + 'attn.Wqkv.weight'].transpose())
    .reshape(b, len, 3, enc.heads, enc.headDim)

  let q = qkv.index(mx.Slice(), mx.Slice(), 0).transpose(0, 2, 1, 3)
  let k = qkv.index(mx.Slice(), mx.Slice(), 1).transpose(0, 2, 1, 3)
  const v = qkv.index(mx.Slice(), mx.Slice(), 2).transpose(0, 2, 1, 3)

  q = mx.fast.rope(q, enc.headDim, false, layer.base, 1.0, 0)
  k = mx.fast.rope(k, enc.headDim, false, layer.base, 1.0, 0)

  const mask = layer.sliding ? localMask : fullMask
  const ctx = mx.fast.scaledDotProductAttention(q, k, v, enc.headDim ** -0.5, mask)
    .transpose(0, 2, 1, 3)
    .reshape(b, len, dims)

  h = mx.add(h, mx.matmul(ctx, w[prefix + 'attn.Wo.weight'].transpose()))

  const mlpNorm = mx.fast.layerNorm(h, w[prefix + 'mlp_norm.weight'], null, enc.normEps)
  const wi = mx.matmul(mlpNorm, w[prefix + 'mlp.Wi.weight'].transpose())
  const [value, gate] = mx.split(wi, 2, -1)

  return mx.add(h, mx.matmul(mx.multiply(gelu(value), gate), w[prefix + 'mlp.Wo.weight'].transpose()))
}

function applyHeadLayer (w, enc, index, h, shape, headMask) {
  const { b, len, dims } = shape
  const prefix = `head.layers.${index}.`

  const normed = mx.fast.layerNorm(h, w[prefix + 'norm1.weight'], w[prefix + 'norm1.bias'], 1e-5)
  const qkv = matmulBias(normed, w[prefix + 'self_attn.in_proj.weight'], w[prefix + 'self_attn.in_proj.bias'])
    .reshape(b, len, 3, enc.headHeads, enc.headHeadDim)

  const q = qkv.index(mx.Slice(), mx.Slice(), 0).transpose(0, 2, 1, 3)
  const k = qkv.index(mx.Slice(), mx.Slice(), 1).transpose(0, 2, 1, 3)
  const v = qkv.index(mx.Slice(), mx.Slice(), 2).transpose(0, 2, 1, 3)

  const ctx = mx.fast.scaledDotProductAttention(q, k, v, enc.headHeadDim ** -0.5, headMask)
    .transpose(0, 2, 1, 3)
    .reshape(b, len, dims)

  h = mx.add(h, matmulBias(ctx, w[prefix + 'self_attn.out_proj.weight'], w[prefix + 'self_attn.out_proj.bias']))

  const inner = mx.fast.layerNorm(h, w[prefix + 'norm2.weight'], w[prefix + 'norm2.bias'], 1e-5)
  const hidden = matmulBias(inner, w[prefix + 'linear1.weight'], w[prefix + 'linear1.bias'])

  return mx.add(h, matmulBias(mx.maximum(hidden, 0), w[prefix + 'linear2.weight'], w[prefix + 'linear2.bias']))
}

function gatherMarkers (h, markerPos, shape, count) {
  const { b, len, dims } = shape

  const rowBase = mx.multiply(mx.expandDims(mx.arange(b, mx.int32), 1), len)
  const positions = mx.add(rowBase, mx.maximum(markerPos, 0)).astype(mx.int32).reshape([b * count])

  return mx.take(h.reshape(-1, dims), positions, 0).reshape(b, count, dims)
}

function scoreMarkers (w, flat, markerMask) {
  let s = mx.fast.layerNorm(flat, w['scorer.layers.0.weight'], w['scorer.layers.0.bias'], 1e-5)
  s = matmulBias(s, w['scorer.layers.1.weight'], w['scorer.layers.1.bias'])
  s = gelu(s)
  s = matmulBias(s, w['scorer.layers.3.weight'], w['scorer.layers.3.bias'])

  return mx.where(
    markerMask,
    s.index(mx.Slice(), mx.Slice(), 0).astype(mx.float32),
    mx.array(-1e4, mx.float32)
  )
}

function agentFeatures (logits, markerMask) {
  const p = mx.softmax(logits, -1)
  const count = mx.maximum(mx.sum(markerMask, -1).astype(mx.float32), 2.0)
  const entropy = mx.divide(
    mx.negative(mx.sum(mx.multiply(p, mx.log(mx.maximum(p, 1e-9))), -1)),
    mx.log(count)
  )

  const top = mx.sort(p, -1).index(mx.Slice(), mx.Slice(-2))
  const top1 = top.index(mx.Slice(), 1)
  const top0 = top.index(mx.Slice(), 0)

  return mx.stack([top1, mx.subtract(top1, top0), entropy, mx.divide(count, 255.0)], -1)
}

function actHead (w, h, features) {
  const pooled = mx.concatenate([h.index(mx.Slice(), 0).astype(mx.float32), features], -1)

  let action = matmulBias(pooled.astype(w['act_head.layers.0.weight'].dtype), w['act_head.layers.0.weight'], w['act_head.layers.0.bias'])
  action = matmulBias(gelu(action), w['act_head.layers.2.weight'], w['act_head.layers.2.bias'])

  return action
}

function encoderPlan (enc) {
  const layerTypes = enc.layer_types || Array.from({ length: enc.num_hidden_layers }, (_, i) =>
    i % (enc.global_attn_every_n_layers || 3) === 0 ? 'full_attention' : 'sliding_attention'
  )
  const params = enc.rope_parameters || {}
  const headHeads = Math.max(1, Math.floor(enc.hidden_size / 64))

  return {
    hiddenSize: enc.hidden_size,
    heads: enc.num_attention_heads,
    headDim: enc.hidden_size / enc.num_attention_heads,
    normEps: enc.norm_eps || 1e-5,
    localAttention: enc.local_attention || 128,
    headLayers: 2,
    headHeads,
    headHeadDim: enc.hidden_size / headHeads,
    layers: layerTypes.map((type, index) => {
      const fallback = type === 'full_attention' ? 160000.0 : 10000.0
      const base = (params[type] || {}).rope_theta || fallback
      return { index, sliding: type === 'sliding_attention', base: Number(base) }
    })
  }
}

function matmulBias (x, weight, bias) {
  const y = mx.matmul(x, weight.transpose())
  return bias ? mx.add(y, bias) : y
}

function gelu (x) {
  return mx.multiply(
    mx.multiply(x, 0.5),
    mx.add(1, mx.erf(mx.multiply(x, Math.SQRT1_2)))
  )
}

function actProbability (logits) {
  const max = Math.max(...logits)
  const exponentials = logits.map(value => Math.exp(value - max))
  const total = exponentials.reduce((sum, value) => sum + value, 0)
  return exponentials[0] / total
}

async function ensureNative () {
  const files = nativeFiles()
  if (files.every(file => fs.existsSync(path.join(NATIVE_DIR, file.name)))) return

  const version = JSON.parse(fs.readFileSync(path.join(PKG_DIR, 'package.json'), 'utf8')).version
  if (version === '0.0.1-dev') return // local development build, binaries are expected in place

  const base = path.join(cacheDir('laya'), 'node-mlx', version)
  fs.mkdirSync(NATIVE_DIR, { recursive: true })

  for (const file of files) {
    const target = path.join(NATIVE_DIR, file.name)
    if (fs.existsSync(target)) continue

    const cached = path.join(base, file.name)
    if (!fs.existsSync(cached)) {
      const url = `https://github.com/frost-beta/node-mlx/releases/download/v${version}/${file.gz}`
      await downloadFile(url, cached, { gunzip: true })
    }
    await fsp.copyFile(cached, target)
  }
}

function nativeFiles () {
  const osName = { darwin: 'mac', win32: 'win' }[process.platform] ?? process.platform
  const prefix = `mlx-${osName}-${process.arch}`

  const files = [{ name: 'node_mlx.node', gz: `${prefix}.node.gz` }]
  if (osName === 'mac' && process.arch === 'arm64') {
    files.push({ name: 'mlx.metallib', gz: `${prefix}.metallib.gz` })
  }

  return files
}
