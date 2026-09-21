const fsp = require('fs/promises')
const path = require('path')
const { Tokenizer } = require('@huggingface/tokenizers')

const QUESTION_TYPES = { choice: 0, score: 1, noul: 2 }

exports.QUESTION_TYPES = QUESTION_TYPES

exports.loadTokenizer = async function (modelDir) {
  const tokenizer = new Tokenizer(
    await readJson(modelDir, 'tokenizer/tokenizer.json'),
    await readJson(modelDir, 'tokenizer/tokenizer_config.json')
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
  const type = question.type
  if (!(type in QUESTION_TYPES)) throw new Error(`Unknown question type ${JSON.stringify(type)}`)

  return {
    t: type,
    ins: renderInstructions(question.instructions),
    crit: normalizeCriteria(type, question.criteria)
  }
}

exports.buildSequence = function (encode, ids, state, question, maxLength, headMaxLength) {
  const scrub = value => value.split(ids.maskToken).join(' ')
  const options = renderOptions(question)

  let optionIds = options.map(option => [ids.mask, ...encode(` ${scrub(option)}`).slice(0, 48)])
  optionIds = fitOptions(optionIds, headMaxLength)

  const budget = headMaxLength - countTokens(optionIds)
  const head = encode(`${question.t} question: ${scrub(question.ins)}`).slice(0, Math.max(8, budget))

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

async function readJson (modelDir, file) {
  const raw = await fsp.readFile(path.join(modelDir, file), 'utf8')
  return JSON.parse(raw)
}

function specialId (tokenizer, token) {
  const id = tokenizer.token_to_id(token)
  if (id === undefined) throw new Error(`Tokenizer is missing ${token}`)
  return id
}

function renderInstructions (instructions) {
  return typeof instructions === 'string' ? instructions : JSON.stringify(instructions)
}

function normalizeCriteria (type, criteria) {
  if (type === 'choice') return normalizeChoiceCriteria(criteria)
  if (type === 'score') return normalizeScoreCriteria(criteria)
  return criteria
}

function normalizeChoiceCriteria (criteria) {
  if (Array.isArray(criteria)) {
    return Object.fromEntries(criteria.map(label => [label, null]))
  }
  if (!criteria || typeof criteria !== 'object') {
    throw new Error('choice criteria must be a nonempty object or array')
  }
  return criteria
}

function normalizeScoreCriteria (criteria) {
  if (!Array.isArray(criteria) || !criteria.length) {
    throw new Error('score criteria must be a nonempty array')
  }
  return criteria
}

function fitOptions (optionIds, headMaxLength) {
  const budget = headMaxLength - countTokens(optionIds)
  if (budget >= 16) return optionIds

  const perOption = Math.max(4, Math.floor((headMaxLength - 16) / Math.max(1, optionIds.length)))
  return optionIds.map(value => value.slice(0, perOption))
}

function countTokens (values) {
  return values.reduce((sum, value) => sum + value.length, 0)
}

function renderOptions (question) {
  if (question.t === 'choice') return renderChoiceOptions(question.crit)
  if (question.t === 'score') return renderScoreOptions(question.crit)
  return renderNoulOptions(question.crit)
}

function renderChoiceOptions (criteria) {
  return Object.entries(criteria).map(([label, value]) =>
    isEmptyCriterion(value) ? label : `${label}: ${renderCriterion(value)}`
  )
}

function renderScoreOptions (criteria) {
  return criteria.map((criterion, i) => `level ${i}: ${renderCriterion(criterion)}`)
}

function renderNoulOptions (criteria) {
  const crit = criteria || {}

  return [
    'false: ' + (isEmptyCriterion(crit.false) ? 'no, the statement does not hold' : renderCriterion(crit.false)),
    'true: ' + (isEmptyCriterion(crit.true) ? 'yes, the statement holds' : renderCriterion(crit.true))
  ]
}

function isEmptyCriterion (value) {
  return value === null || value === undefined || value === ''
}

function renderCriterion (value) {
  return typeof value === 'string' ? value : pyJsonDumps(value)
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
