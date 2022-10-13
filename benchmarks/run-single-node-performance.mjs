#!/usr/bin/env node

import { createReadStream, createWriteStream } from 'node:fs';
import { readdir, mkdir, readFile, rename } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { parse as parseCsv, format as formatCsv } from 'fast-csv';

const workingDirectory = new URL('./', import.meta.url);

const currentDate = new Date().toISOString().replace(/\..*/, '').replace(/:/g, '-').replace('T', '_');
const outputDirectory = new URL(`./output/${currentDate}_single_node_performance/`, workingDirectory);
const csvFile = new URL('./single_node_performance.csv', outputDirectory);
const csvFileWithPapi = new URL('./single_node_performance_with_papi.csv', outputDirectory);

await mkdir(outputDirectory, { recursive: true });

const childProcess = spawn('./bin/single_node_performance.exe', {
  cwd: workingDirectory,
  shell: true,
});

let writeStream = createWriteStream(csvFile);
childProcess.stdout.pipe(writeStream);
childProcess.stdout.pipe(process.stdout);

childProcess.on('close', async () => {
  writeStream.close();

  const papiFolder = new URL('./papi_hl_output/', import.meta.url);
  const papiFiles = await readdir(papiFolder);

  const papiJsonFile = new URL(papiFiles[0], papiFolder);
  const papiData = JSON.parse(await readFile(papiJsonFile));

  await rename(papiFolder, new URL('./papi_hl_output/', outputDirectory));

  writeStream = createWriteStream(csvFileWithPapi);
  let rowCount = 0;

  createReadStream(csvFile)
    .pipe(parseCsv({
      headers: true,
      trim: true,
    }))
    .pipe(formatCsv({
      headers: true,
      transform: row => ({
        ...row,
        ...getPapiRowData(papiData, rowCount++),
      }),
    }))
    .pipe(writeStream);
});

const getPapiRowData = (papiData, rowCount) => Object.fromEntries(
  Object.entries(papiData.threads['0'].regions[String(rowCount)])
    .filter(([key]) => key !== 'name' && key !== 'parent_region_id')
    .map(([key, value]) => [key.trim().replace(/[:-]/g, '_'), value]),
);