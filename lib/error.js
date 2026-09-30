module.exports = class ErrorJEV extends Error {
  constructor (message, code, cause) {
    super(code + ': ' + message)

    this.code = code

    if (cause) {
      this.cause = cause
    }
  }

  get name () {
    return 'ErrorJEV'
  }
}
