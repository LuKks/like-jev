'use strict'

const fsp = require('fs/promises')
const path = require('path')
const { Tokenizer } = require('@huggingface/tokenizers')

const QUESTION_TYPES = { choice: 0, score: 1, noul: 2 }

exports.QUESTION_TYPES = QUESTION_TYPES

exports.loadTokenizer = async function (modelDir) {
  const readJson = file => fsp.readFile(path.join(modelDir, file), 'utf8').then(JSON.parse)
  const tokenizer = new Tokenizer(
    await readJson('tokenizer/tokenizer.json'),
    await readJson('tokenizer/tokenizer_config.json')
  )
  return {
    tokenizer,
    ids: {
      cls: specialId(tokenizer, '[CLS]'),
      sep: specialId(tokenizer, '[SEP]'),
      mask: specialId(tokenizer, '[MASK]'),
      pad: specialId(tokenizer, '[PAD]'),
      maskToken: '[MASK]'
    }
  }
}

exports.toInternal = function (question) {
  const t = question.type
  if (!(t in QUESTION_TYPES)) throw new Error(`Unknown question type ${JSON.stringify(t)}`)
  let criteria = question.criteria
  if (t === 'choice') {
    if (Array.isArray(criteria)) criteria = Object.fromEntries(criteria.map(label => [label, null]))
    if (!criteria || typeof criteria !== 'object') throw new Error('choice criteria must be a nonempty object or array')
  } else if (t === 'score') {
    if (!Array.isArray(criteria) || !criteria.length) throw new Error('score criteria must be a nonempty array')
  }
  return {
    t,
    ins: typeof question.instructions === 'string' ? question.instructions : JSON.stringify(question.instructions),
    crit: criteria
  }
}

exports.buildSequence = function (encode, ids, state, question, maxLength, headMaxLength) {
  const scrub = value => value.split(ids.maskToken).join(' ')
  const options = renderOptions(question)
  let head = encode(`${question.t} question: ${scrub(question.ins)}`)
  let optionIds = options.map(option => [ids.mask, ...encode(` ${scrub(option)}`).slice(0, 48)])
  const total = values => values.reduce((sum, value) => sum + value.length, 0)
  let optionBudget = headMaxLength - total(optionIds)

  if (optionBudget < 16) {
    const perOption = Math.max(4, Math.floor((headMaxLength - 16) / Math.max(1, optionIds.length)))
    optionIds = optionIds.map(value => value.slice(0, perOption))
    optionBudget = headMaxLength - total(optionIds)
  }

  head = head.slice(0, Math.max(8, optionBudget))
  const sequence = [ids.cls, ...head, ids.sep]
  const markers = []
  for (const option of optionIds) {
    markers.push(sequence.length)
    sequence.push(...option)
  }
  sequence.push(ids.sep)

  const room = Math.max(0, maxLength - sequence.length - 1)
  const stateIds = encode(scrub(serializeState(state))).slice(0, room)
  sequence.push(...stateIds, ids.sep)

  return {
    ids: sequence.slice(0, maxLength),
    markers: markers.filter(marker => marker < maxLength),
    options
  }
}

function specialId (tokenizer, token) {
  const id = tokenizer.token_to_id(token)
  if (id === undefined) throw new Error(`Tokenizer is missing ${token}`)
  return id
}

function renderCriterion (value) {
  return typeof value === 'string' ? value : pyJsonDumps(value)
}

function renderOptions (question) {
  if (question.t === 'choice') {
    return Object.entries(question.crit).map(([label, value]) =>
      value === null || value === undefined || value === '' ? label : `${label}: ${renderCriterion(value)}`
    )
  }
  if (question.t === 'score') {
    return question.crit.map((criterion, i) => `level ${i}: ${renderCriterion(criterion)}`)
  }
  const criteria = question.crit || {}
  const noFalse = criteria.false === null || criteria.false === undefined || criteria.false === ''
  const noTrue = criteria.true === null || criteria.true === undefined || criteria.true === ''
  return [
    'false: ' + (noFalse ? 'no, the statement does not hold' : renderCriterion(criteria.false)),
    'true: ' + (noTrue ? 'yes, the statement holds' : renderCriterion(criteria.true))
  ]
}

function serializeState (state) {
  return typeof state === 'string' ? state : pyJsonDumps(state)
}

function pyJsonDumps (value) {
  if (value === null || value === undefined) return 'null'
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : JSON.stringify(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (Array.isArray(value)) return '[' + value.map(pyJsonDumps).join(', ') + ']'
  return '{' + Object.entries(value).map(([key, item]) => `${JSON.stringify(key)}: ${pyJsonDumps(item)}`).join(', ') + '}'
}
