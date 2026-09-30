// AI

const { QUESTION_TYPES, buildSequence, toInternal } = require('./lib/questions.js')
const { decodeAnswers } = require('./lib/decode.js')
const { makeBatch } = require('./lib/batch.js')
const ErrorJEV = require('./lib/error.js')

const BACKENDS = ['auto', 'onnx', 'coreml']

module.exports = class Jev {
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
    if (this._closed) throw new Error('jev is closed')

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
    if (typeof state !== 'string' && (state === null || typeof state !== 'object')) {
      throw new ErrorJEV('state must be a string, object, or array', 'INVALID_STATE')
    }

    await this.ready()

    const questionIds = Object.keys(questions)
    if (!questionIds.length) {
      throw new ErrorJEV('ask: at least one question is required', 'QUESTIONS_REQUIRED')
    }

    const items = this._buildItems(state, questions, questionIds)
    const batch = makeBatch(items, this.ids.pad, this.config.max_len, this._engine.padToMultiple)
    const { logits } = await this._engine.forward(batch)

    return {
      model: `jev-${this._engine.backend}-${this._engine.model}`,
      answers: decodeAnswers(questionIds, items, logits, this.config),
      usage: {
        input_tokens: items.reduce((sum, item) => sum + item.ids.length, 0),
        output_tokens: 0
      }
    }
  }

  async noul (state, instructions, criteria) {
    const response = await this.ask(state, {
      answer: { type: 'noul', instructions, criteria }
    })
    return response.answers.answer
  }

  async choice (state, instructions, criteria) {
    const response = await this.ask(state, {
      answer: { type: 'choice', instructions, criteria }
    })
    return response.answers.answer
  }

  async score (state, instructions, criteria) {
    const response = await this.ask(state, {
      answer: { type: 'score', instructions, criteria }
    })
    return response.answers.answer
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

  _buildItems (state, questions, questionIds) {
    const encode = text => this.tokenizer.encode(text, { add_special_tokens: false }).ids

    return questionIds.map(qid => {
      const question = toInternal(questions[qid])

      let sequence
      try {
        sequence = buildSequence(encode, this.ids, state, question, this.config.max_len, this.config.head_max_len)
      } catch (err) {
        const message = err.code && err.message.startsWith(`${err.code}: `)
          ? err.message.slice(err.code.length + 2)
          : err.message

        if (err.code) {
          throw new ErrorJEV(`question ${JSON.stringify(qid)}: ${message}`, err.code, err)
        }

        const wrapped = new Error(`question ${JSON.stringify(qid)}: ${message}`)
        wrapped.cause = err
        throw wrapped
      }

      if (sequence.markers.length !== sequence.options.length) {
        throw new ErrorJEV(`question ${JSON.stringify(qid)}: options do not fit in head_max_len=${this.config.head_max_len} tokens`, 'HEAD_TOO_LONG')
      }

      return { question, ...sequence, qtype: QUESTION_TYPES[question.t] }
    })
  }
}

async function loadEngine (opts) {
  const backend = opts.backend || 'auto'
  if (!BACKENDS.includes(backend)) throw new Error(`backend must be one of ${BACKENDS}`)

  const picked = backend === 'auto'
    ? (process.platform === 'darwin' ? 'coreml' : 'onnx')
    : backend
  if (picked === 'coreml' && process.platform !== 'darwin') {
    throw new Error('Core ML requires macOS')
  }

  return require(`./lib/${picked}.js`).load(opts)
}
