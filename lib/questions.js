const ErrorJEV = require('./error.js')

const QUESTION_TYPES = { choice: 0, score: 1, noul: 2 }
const OPTION_MAX_TOKENS = 48
const HEAD_RESERVED_TOKENS = 16

exports.QUESTION_TYPES = QUESTION_TYPES

exports.toInternal = function (question) {
  const type = question.type
  if (!(type in QUESTION_TYPES)) {
    throw codedError('UNKNOWN_QUESTION_TYPE', `Unknown question type ${JSON.stringify(type)}`)
  }

  return {
    t: type,
    ins: renderInstructions(question.instructions),
    crit: normalizeCriteria(type, question.criteria)
  }
}

exports.prepareState = function (encode, ids, state) {
  return encode(scrubText(serializeState(state), ids.maskToken))
}

exports.buildSequence = function (encode, ids, state, question, maxLength, headMaxLength, preparedState) {
  const options = renderOptions(question)

  const optionIds = options.map(option => [ids.mask, ...encode(` ${scrubText(option, ids.maskToken)}`)])

  optionIds.forEach((option, i) => {
    const length = option.length - 1
    if (length > OPTION_MAX_TOKENS) {
      throw codedError('OPTION_TOO_LONG', `option ${i + 1} is ${length} tokens long, limit is ${OPTION_MAX_TOKENS} tokens`)
    }
  })

  const optionsTokens = countTokens(optionIds)
  const budget = headMaxLength - optionsTokens
  if (budget < HEAD_RESERVED_TOKENS) {
    throw codedError('HEAD_TOO_LONG', `options need ${optionsTokens} tokens, head_max_len is ${headMaxLength} tokens`)
  }

  const instructions = encode(`${question.t} question: ${scrubText(question.ins, ids.maskToken)}`)
  if (instructions.length > budget) {
    throw codedError('INSTRUCTIONS_TOO_LONG', `instructions are ${instructions.length} tokens long, limit is ${budget} tokens`)
  }

  const sequence = [ids.cls, ...instructions, ids.sep]
  const markers = []
  for (const option of optionIds) {
    markers.push(sequence.length)
    sequence.push(...option)
  }
  sequence.push(ids.sep)

  const room = maxLength - sequence.length - 1
  const stateIds = preparedState || encode(scrubText(serializeState(state), ids.maskToken))
  if (stateIds.length > room) {
    throw codedError('STATE_TOO_LONG', `state is ${stateIds.length} tokens long, limit is ${room} tokens`)
  }
  sequence.push(...stateIds, ids.sep)

  return {
    ids: sequence,
    markers,
    options
  }
}

function scrubText (value, maskToken) {
  return value.split(maskToken).join(' ')
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
  if (!criteria || typeof criteria !== 'object' || Array.isArray(criteria) || !Object.keys(criteria).length) {
    throw codedError('INVALID_CHOICE_CRITERIA', 'choice criteria must be a nonempty object')
  }
  return criteria
}

function normalizeScoreCriteria (criteria) {
  if (!Array.isArray(criteria) || !criteria.length) {
    throw codedError('INVALID_SCORE_CRITERIA', 'score criteria must be a nonempty array')
  }
  return criteria
}

function countTokens (values) {
  return values.reduce((sum, value) => sum + value.length, 0)
}

function codedError (code, message) {
  return new ErrorJEV(message, code)
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
