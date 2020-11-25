# DeepEq

The primary goal of DeepEq is to create a bridge between deep learning theory and practice and allow users to reason about the implications of backpropagation in an interactive way. The equations are kept as simple as possible to make them accessible to high school level students and above. DeepEq provides the following functionality:

- It allows users to create a small artificial neural network by joining perceptrons
- A simple ”click” on a perceptron reveals the underlying (colorized) equations for forward- and backpropagation
- The integrated code editor allows users to tinker with and implement their own backpropagation algorithm
- A complementary instructional tutorial that serves as a guide throughout the learning process

# Overview

This repo was created using [Idyll](https://idyll-lang.org) and [Math.js](https://mathjs.org).
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

## Developing locally

Run `idyll`.  
(Make sure to have all dependencies installed: `npm install`)

## Building for production

Run `idyll build`. The output will appear in the top-level `build` folder.  
(For Github-Pages: Copy the contents of the `build` folder to the `docs` folder)

## Deploying

Make sure your post has been built, then deploy the docs folder via any static hosting service.

## Dependencies

You can install custom dependencies by running `npm install <package-name> --save`. Note that any collaborators will also need download the package locally by running `npm install` after pulling the changes.
