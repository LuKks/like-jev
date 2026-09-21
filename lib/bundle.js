const fs = require('fs')
const fsp = require('fs/promises')
const os = require('os')
const path = require('path')
const zlib = require('zlib')
const { Readable } = require('stream')
const { pipeline } = require('stream/promises')

exports.cacheDir = function (name) {
  if (process.platform === 'darwin') {
    return path.join(os.homedir(), 'Library', 'Caches', name)
  }

  if (process.platform === 'win32') {
    const base = process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local')
    return path.join(base, name, 'Cache')
  }

  return path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache'), name)
}

exports.ensureBundle = async function (opts) {
  if (bundleComplete(opts.files, opts.cacheDir)) return opts.cacheDir
  if (opts.fallbackDir && bundleComplete(opts.files, opts.fallbackDir)) return opts.fallbackDir

  await downloadBundle(opts)
  return opts.cacheDir
}

exports.bundleComplete = bundleComplete
exports.downloadFile = downloadFile

function bundleComplete (files, dir) {
  if (!dir) return false
  return files.every(file => fs.existsSync(path.join(dir, file)))
}

async function downloadBundle (opts) {
  await fsp.mkdir(opts.cacheDir, { recursive: true })

  for (const file of opts.files) {
    const target = path.join(opts.cacheDir, file)
    if (fs.existsSync(target)) continue

    await downloadFile(fileUrl(opts.base, file), target)
  }
}

function fileUrl (base, file) {
  const encoded = file.split('/').map(encodeURIComponent).join('/')
  return `${base}/${encoded}`
}

async function downloadFile (url, target, opts = {}) {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Could not download ${url}: ${response.status} ${response.statusText}`)
  }

  fs.mkdirSync(path.dirname(target), { recursive: true })

  const temporary = target + '.part'
  const chain = [Readable.fromWeb(response.body)]
  if (opts.gunzip) chain.push(zlib.createGunzip())
  chain.push(fs.createWriteStream(temporary))

  await pipeline(...chain)
  await fsp.rename(temporary, target)
}
