'use strict'

const QUESTION_NAMES = ['choice', 'score', 'noul']

exports.decodeAnswers = function (questionIds, items, logits, actProbabilities, config) {
  const answers = {}

  items.forEach((item, row) => {
    const key = questionIds[row]
    const probabilities = optionProbabilities(item, logits[row], config)
    const agent = { act_probability: round4(actProbabilities[row][0]) }

    answers[key] = buildAnswer(item.question, probabilities, agent)
  })

  return answers
}

function optionProbabilities (item, rowLogits, config) {
  const count = item.markers.length
  const bucket = tempBucket(item.qtype, count)
  const temperature = config.temperature_by_options[bucket] ?? config.temperature[item.qtype] ?? 1
  const scaled = rowLogits.slice(0, count).map(value => value / Math.max(1e-3, temperature))

  return softmax(scaled)
}

function buildAnswer (question, probabilities, agent) {
  if (question.t === 'choice') return choiceAnswer(question.crit, probabilities, agent)
  if (question.t === 'score') return scoreAnswer(question.crit, probabilities, agent)
  return noulAnswer(probabilities, agent)
}

function choiceAnswer (criteria, p, agent) {
  const labels = Object.keys(criteria)
  const best = p.indexOf(Math.max(...p))

  return {
    type: 'choice',
    choice: labels[best],
    probabilities: Object.fromEntries(labels.map((label, i) => [label, round4(p[i])])),
    confidence: round4(confidence(p)),
    rl_agent: agent
  }
}

function scoreAnswer (criteria, p, agent) {
  return {
    type: 'score',
    score: round4(p.reduce((sum, value, i) => sum + i * value, 0)),
    legend: Object.fromEntries(criteria.map((criterion, i) => [String(i), criterion])),
    probabilities: Object.fromEntries(p.map((value, i) => [String(i), round4(value)])),
    confidence: round4(confidence(p)),
    rl_agent: agent
  }
}

function noulAnswer (p, agent) {
  return {
    type: 'noul',
    noul: round4(p[1]),
    confidence: round4(Math.max(p[1], 1 - p[1])),
    rl_agent: agent
  }
}

function tempBucket (qtype, count) {
  let size = '11+'
  if (count <= 2) size = '2'
  else if (count <= 5) size = '3-5'
  else if (count <= 10) size = '6-10'

  return QUESTION_NAMES[qtype] + ':' + size
}

function softmax (values) {
  const max = Math.max(...values)
  const exponentials = values.map(value => Math.exp(value - max))
  const total = exponentials.reduce((sum, value) => sum + value, 0)

  return exponentials.map(value => value / total)
}

function confidence (probabilities) {
  if (probabilities.length < 2) return 1

  const entropy = probabilities.reduce((sum, value) => sum - value * Math.log(Math.max(value, 1e-12)), 0)
  return Math.min(1, Math.max(0, 1 - entropy / Math.log(probabilities.length)))
}

function round4 (value) {
  return Math.round(value * 1e4) / 1e4
}
