'use strict'

const { QUESTION_TYPES, buildSequence, toInternal } = require('./lib/questions.js')
const { decodeAnswers } = require('./lib/decode.js')
const { makeBatch } = require('./lib/batch.js')

const BACKENDS = ['auto', 'onnx', 'mlx']

class Laya {
  constructor (opts = {}) {
    this._opts = opts
    this._engine = null
    this._closed = false
    this.opened = false
    this.opening = this.ready()
    this.opening.then(() => {
      this.opened = true
    }).catch(() => {})
  }

  async ready () {
    if (this.opened) return
    if (this.opening) return this.opening
    if (this._closed) throw new Error('laya is closed')

    try {
      const engine = this._opts.engine || await loadEngine(this._opts)
      this._engine = engine
      this.tokenizer = engine.tokenizer
      this.ids = engine.ids
      this.config = engine.config
      this.modelDir = engine.modelDir
    } catch (err) {
      this.opening = null
      throw err
    }
  }

  async ask (state, questions) {
    await this.ready()

    const questionIds = Object.keys(questions)
    if (!questionIds.length) throw new Error('ask: at least one question is required')

    const encode = text => this.tokenizer.encode(text, { add_special_tokens: false }).ids
    const items = questionIds.map(qid => {
      const question = toInternal(questions[qid])
      const sequence = buildSequence(encode, this.ids, state, question, this.config.max_len, this.config.head_max_len)
      if (sequence.markers.length !== sequence.options.length) {
        throw new Error(`question ${JSON.stringify(qid)}: options do not fit in head_max_len=${this.config.head_max_len} tokens`)
      }
      return { question, ...sequence, qtype: QUESTION_TYPES[question.t] }
    })

    const batch = makeBatch(items, this.ids.pad, this.config.max_len, this._engine.padToMultiple)
    const { logits, actProbabilities } = await this._engine.forward(batch)

    return {
      model: this._engine.name,
      answers: decodeAnswers(questionIds, items, logits, actProbabilities, this.config),
      usage: { input_tokens: items.reduce((sum, item) => sum + item.ids.length, 0), output_tokens: 0 }
    }
  }

  async systemOne (state, questions) {
    return this.ask(state, questions)
  }

  async close () {
    if (this._closed) return
    this._closed = true
    if (this.opening) {
      try {
        await this.opening
      } catch {}
    }
    if (this._engine) await this._engine.close()
    this.opened = false
  }
}

async function loadEngine (opts) {
  const backend = opts.backend || 'auto'
  if (!BACKENDS.includes(backend)) throw new Error(`backend must be one of ${BACKENDS}`)
  const picked = backend === 'auto'
    ? (process.platform === 'darwin' ? 'mlx' : 'onnx')
    : backend
  return require(`./lib/${picked}.js`).load(opts)
}

module.exports = Laya
module.exports.Laya = Laya
