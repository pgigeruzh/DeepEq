# Overview

This repo was created using [Idyll](https://idyll-lang.org), [Math.js](https://mathjs.org), and [TensorFlow.js](https://www.tensorflow.org/js).
It contains the following files and directories (only the most important are listed here):

| Directory  | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| index.idyll| Entrypoint file                                              |
| styles.css | CSS stylesheet                                               |
| components | Custom react components (flowchart.js is the main component) |
| static     | Static files such as images                                  |
| docs       | A copy of the build folder (for hosting on GitHub Pages)     |

## Installation

- Make sure you have `idyll` installed (`npm i -g idyll`).
- Clone this repo and run `npm install`.

## Developing a post locally

Run `idyll`.

## Building a post for production

Run `idyll build`. The output will appear in the top-level `build` folder. To change the output location, change the `output` option in `package.json`.

## Deploying

Make sure your post has been built, then deploy the docs folder via any static hosting service.

## Dependencies

You can install custom dependencies by running `npm install <package-name> --save`. Note that any collaborators will also need download the package locally by running `npm install` after pulling the changes.
