import React, { useState, useEffect } from 'react';
import ReactFlow, { ReactFlowProvider, removeElements, addEdge, Controls } from 'react-flow-renderer';
import MathJax from 'react-mathjax2';
import { GithubPicker } from 'react-color'
import { VictoryChart, VictoryAxis, VictoryLine, VictoryScatter } from 'victory';
import Editor from 'react-simple-code-editor';
import Prism from "prismjs";
const { filterChildren, mapChildren } = require('idyll-component-children');
const AriaModal = require('react-aria-modal');
const math = require('mathjs');

const initialElements = [
    { id: 'x0', data: { label: 'x0: 0.1, \u0278(x)=x', value: 0.1, activation: 'x', color: '#4A90E2' }, position: { x: 200, y: 200 }, sourcePosition: 'right', type: 'input' },
    { id: 'x1', data: { label: 'x1: 0.3, \u0278(x)=x', value: 0.3, activation: 'x', color: '#4A90E2' }, position: { x: 200, y: 400 }, sourcePosition: 'right', type: 'input' },
    { id: 'n0', data: { label: 'n0: 0.0, \u0278(x)=x', value: 0.0, activation: 'tanh(x)', color: '#4A4A4A' }, position: { x: 550, y: 250 }, targetPosition: 'left', sourcePosition: 'right' },
    { id: 'n1', data: { label: 'n1: 0.0, \u0278(x)=x', value: 0.0, activation: 'cos(x)', color: '#4A4A4A' }, position: { x: 550, y: 350 }, targetPosition: 'left', sourcePosition: 'right' },
    { id: 'y0', data: { label: 'y0: 0.0, \u0278(x)=x', value: 0.0, activation: '1/(1+e^(-x))', true: 0.4, color: '#D0021B' }, position: { x: 800, y: 300 }, targetPosition: 'left', type: 'output' },
    { source: "x0", target: "n0", id: "w0", label: "w0: 0.1", data: { value: 0.1, color: '#4A90E2' } },
    { source: "x0", target: "n1", id: "w1", label: "w1: 0.2", data: { value: 0.2, color: '#4A90E2' } },
    { source: "x1", target: "n0", id: "w2", label: "w2: 0.3", data: { value: 0.3, color: '#4A90E2' } },
    { source: "x1", target: "n1", id: "w3", label: "w3: 0.4", data: { value: 0.4, color: '#4A90E2' } },
    { source: "n0", target: "y0", id: "w4", label: "w4: 0.5", data: { value: 0.5, color: '#4A4A4A' } },
    { source: "n1", target: "y0", id: "w5", label: "w6: 0.6", data: { value: 0.6, color: '#4A4A4A' } },
];

let completeBackpropagationAlgorithm = `
// run forward propagation for each output node
let forwardresults = [];
let outputNodes = graph.filter(element => element.id.includes("y"));
for (let i in outputNodes) { forwardpropagation(graph, outputNodes[i], forwardresults); }

// run backpropagation for each input node
let backpropresults = [];
let inputNodes = graph.filter(element => element.id.includes("x"));
for (let i in inputNodes) { backpropagation(graph, inputNodes[i], forwardresults, backpropresults); }

// merge and assign results
for (let i = 0; i < backpropresults.length; i++) {
  results.push({
    ...backpropresults[i],
    ...(forwardresults.find((e) => e.id == backpropresults[i].id))
  });
}

// calculate and assign loss
let loss = 0.0;
for (let i in outputNodes) {
  let nodes = forwardresults.filter(element => element.id.includes(outputNodes[i].id));
  if (nodes.length > 0) {
    let t = outputNodes[i].data.true;
    let y = nodes[0].value;
    loss += 0.5 * (t - y) ** 2;
  }
}
results.push({ loss: loss });

// recursive forwardpropagation algorithm
function forwardpropagation(graph, node, results) {
  // get all ingoing edges from node
  let edges = graph.filter(element => element.target == node.id);
  if (edges.length > 0) {
    let sum = 0;
    let symbolicEquation = "";
    let symbolicSum = "";
    for (let j in edges) {
      let edge = edges[j];
      let nextNode = graph.filter(element => element.id == edge.source)[0];
      // recursive call
      let [nextNodeValue, nextNodeSymbolicEquation] = forwardpropagation(graph, nextNode, results);
      // weighted sum of perceptron e.g. x1*w1+...+xn*wn
      sum += edge.data.value * nextNodeValue;
      symbolicEquation += "<div style='color:" + edge.data.color + ";display:inline;'>("
                         + nextNodeSymbolicEquation + "*" + edge.id + ")</div>";
      symbolicSum += "<div style='color:" + edge.data.color + ";display:inline;'>("
                         + nextNode.id + "*" + edge.id + ")</div>";
      if (j < edges.length - 1) { symbolicEquation += "+"; symbolicSum += "+" };
    }
    // activation function e.g. a(x1*w1+...+xn*wn)
    let value = math.evaluate(node.data.activation.replace('x', sum));
    let activation = node.data.activation;
    if ((activation[activation.indexOf('x') - 1] != '(')
        || (activation[activation.indexOf('x') + 1] != ')'))
    {
      activation = activation.replace('x', "(x)")
    }
    symbolicEquation = "<div style='color:" + node.data.color + ";display:inline;'>"
                        + activation.replace('x', symbolicEquation) + "</div>";
    // save result
    let result = { id: node.id, sum: sum, symbolicSum: symbolicSum,
                    value: value, symbolicEquation: symbolicEquation, color: node.data.color };
    results.push(result);
    // return values for the next recursive call
    return [value, symbolicEquation];
  } else if (node.id.includes("x")) {
    // stop recursion when reaching an input node
    let result = { id: node.id, value: node.data.value, symbolicEquation: node.id, color: node.data.color };
    results.push(result);
    return [node.data.value, node.id];
  } else {
    // stop recursion when reaching a node without connections
    return NaN;
  }
}

// recursive backpropagation algorithm
function backpropagation(graph, node, forwardresults, results) {
  // get all outgoing edges from node
  let edges = graph.filter(element => element.source == node.id);
  if (edges.length > 0) {
    let totalGradient = 0.0;
    let totalSymbolicGradient = "";
    for (let j in edges) {
      let edge = edges[j];
      let nextNode = graph.filter(element => element.id == edge.target)[0];
      // recursive call
      let [nextNodeGradient, nextSymbolicGradient] = backpropagation(graph, nextNode, forwardresults, results);
      // use latest data from forward propagation
      let tmp = forwardresults.filter(element => element.id == node.id)[0];
      let nodeColor = tmp.color;
      let nodeSymbolicSum = tmp.symbolicSum;
      let nodeSum = tmp.sum;
      let nodeValue = tmp.value;
      // calculate gradient
      let outerDerivative = math.derivative(node.data.activation, 'x');
      totalGradient += nextNodeGradient * outerDerivative.evaluate({ x: nodeSum }) * edge.data.value;
      totalSymbolicGradient += "<div style='color:" + nextNode.data.color + ";display:inline;'>"
                            + nextSymbolicGradient + "</div>" + "*(" + "<div style='color:"
                            + node.data.color + ";display:inline;'>"
                            + outerDerivative.toString().replace(/ /g,'').replace('x', nodeSymbolicSum)
                            + "</div>" + "*"
                            + "<div style='color:" + node.data.color + ";display:inline;'>" + edge.id
                            + "</div>" + ")";
      if (j < edges.length - 1) totalSymbolicGradient += "+";
      // save result for edge
      let edgeGradient = {
        id: edge.id,
        gradient: nextNodeGradient * nodeValue,
        symbolicGradient: "<div style='color:" + nextNode.data.color + ";display:inline;'>"
                          + nextSymbolicGradient + "</div>" + "*" + "<div style='color:"
                          + node.data.color + ";display:inline;'>" + node.id + "</div>",
        color: nodeColor
      };
      results.push(edgeGradient);
    }
    // save result for node
    let nodeGradient = { id: node.id, gradient: totalGradient, symbolicGradient: totalSymbolicGradient };
    results.push(nodeGradient);
    // return values for the next recursive call
    return [totalGradient, node.id + "'"];
  } else if (node.id.includes("y")) {
    // stop recursion when reaching an output node
    let tmp = forwardresults.filter(element => element.id == node.id)[0];
    let nodeSum = tmp.sum;
    let nodeSymbolicSum = tmp.symbolicSum;
    let nodeValue = tmp.value;
    let outerDerivative = math.derivative(node.data.activation, 'x');
    // save result
    let nodeGradient = {
      id: node.id,
      gradient: -1 * (node.data.true - nodeValue) * outerDerivative.evaluate({ x: nodeSum }),
      symbolicGradient: "<div style='color:" + node.data.color + ";display:inline;'>"
                        + "-1*(t" + node.id.substring(1) + "-" + node.id + ")*"
                        + outerDerivative.toString().replace(/ /g,'').replace('x', nodeSymbolicSum)
                        + "</div>"
    };
    results.push(nodeGradient);
    return [nodeGradient.gradient, node.id + "'"];
  } else {
    // stop recursion when reaching a node without connections
    return NaN;
  }
}
`;
let minimalBackpropagationAlgorithm = `
// run forward propagation for each output node
let forwardresults = [];
let outputNodes = graph.filter(element => element.id.includes("y"));
for (let i in outputNodes) { forwardpropagation(graph, outputNodes[i], forwardresults); }

// run backpropagation for each input node
let backpropresults = [];
let inputNodes = graph.filter(element => element.id.includes("x"));
for (let i in inputNodes) { backpropagation(graph, inputNodes[i], forwardresults, backpropresults); }

// merge and assign results
for (let i = 0; i < backpropresults.length; i++) {
  results.push({
    ...backpropresults[i],
    ...(forwardresults.find((e) => e.id == backpropresults[i].id))
  });
}

// recursive forwardpropagation algorithm
function forwardpropagation(graph, node, results) {
  // get all ingoing edges from node
  let edges = graph.filter(element => element.target == node.id);
  if (edges.length > 0) {
    let sum = 0;
    for (let j in edges) {
      let edge = edges[j];
      let nextNode = graph.filter(element => element.id == edge.source)[0];
      // recursive call
      let nextNodeValue = forwardpropagation(graph, nextNode, results);
      // calculate weighted sum of perceptron e.g. x1*w1+...+xn*wn
      sum += edge.data.value * nextNodeValue;
    }
    // apply activation function e.g. a(x1*w1+...+xn*wn)
    let value = math.evaluate(node.data.activation.replace('x', sum));
    // save result
    let result = { id: node.id, sum: sum, value: value };
    results.push(result);
    // return value for the next recursive call
    return value;
  } else if (node.id.includes("x")) {
    // stop recursion when reaching an input node
    let result = { id: node.id, value: node.data.value };
    results.push(result);
    return node.data.value;
  } else {
    // stop recursion when reaching a node without connections
    return NaN;
  }
}

// recursive backpropagation algorithm
function backpropagation(graph, node, forwardresults, results) {
  // get all outgoing edges from node
  let edges = graph.filter(element => element.source == node.id);
  if (edges.length > 0) {
    let totalGradient = 0.0;
    for (let j in edges) {
      let edge = edges[j];
      let nextNode = graph.filter(element => element.id == edge.target)[0];
      // recursive call
      let nextNodeGradient = backpropagation(graph, nextNode, forwardresults, results);
      // use latest data from forward propagation
      let tmp = forwardresults.filter(element => element.id == node.id)[0];
      let nodeSum = tmp.sum;
      let nodeValue = tmp.value;
      // calculate gradient
      let outerDerivative = math.derivative(node.data.activation, 'x');
      totalGradient += nextNodeGradient * outerDerivative.evaluate({ x: nodeSum }) * edge.data.value;
      // save result for edge
      let edgeGradient = { id: edge.id, gradient: nextNodeGradient * nodeValue };
      results.push(edgeGradient);
    }
    // save result for node
    let nodeGradient = { id: node.id, gradient: totalGradient };
    results.push(nodeGradient);
    // return values for the next recursive call
    return totalGradient;
  } else if (node.id.includes("y")) {
    // stop recursion when reaching an output node
    let tmp = forwardresults.filter(element => element.id == node.id)[0];
    let nodeSum = tmp.sum;
    let nodeValue = tmp.value;
    let outerDerivative = math.derivative(node.data.activation, 'x');
    // save result
    let nodeGradient = {
      id: node.id,
      gradient: -1 * (node.data.true - nodeValue) * outerDerivative.evaluate({ x: nodeSum }),
    };
    results.push(nodeGradient);
    return nodeGradient.gradient;
  } else {
    // stop recursion when reaching a node without connections
    return NaN;
  }
}
`
let initialCustomBackpropagationAlgorithm = `
/** The following functions and variables are available to you

 * graph
  Description: "An array of objects containing the data of all nodes and edges"
  Note: Use the filter function to traverse the graph e.g. graph.filter(element => element.target == "n0")
  Values Node:
   - id: the id of the node
   - data.activation: a string of the activation function e.g. tanh(x)
   - data.sum: value assigned by you, see "result"
   - data.value: value assigned by you, see "result"
   - data.gradient: value assigned by you, see "result"
   - data.symbolicEquation: value assigned by you, see "result"
   - data.symbolicGradient: value assigned by you, see "result"
  Example Node:
   {id: "n0", data: {sum: 0.1, value: 0.0997, gradient: 0.0285, activation: "tanh(x)" }}
  Values Edge:
   - id: id of the edge
   - source: id of the starting point of the edge
   - target: if of the end point of the edge
   - data.value: the value of the weight/edge
   - data.gradient: value assigned by you, see "result"
   - data.symbolicEquation: value assigned by you, see "result"
   - data.symbolicGradient: value assigned by you, see "result"
  Example Edge:
   {id: "w2", source: "x0", target: "n1", data: {value: 0.2, gradient: -0.0005 }}

 * math
  Description: "A reference to math.js, an extensive math library for JavaScript"
  Example: derivative of tanh: math.derivative('tanh(x)', 'x');

 * result
  Description: "An empty array of objects to be filled with your results"
  Note: You do not need to use assign variables
  Values:
   - id: the id of the node or edge (mandatory)
   - sum: the weighted sum e.g. x1*w1+...+xn*wn (optional)
   - value: value after the activation function e.g. a(x1*w1+...+xn*wn) (optional)
   - gradient: the value of the calculated gradient (optional)
   - symbolicEquation: the mathematical formula of the forwardpropagation pass (optional)
   - symbolicGradient: the mathematical formula of the backpropagation pass (optional)
  Example:
   let result =
   {
     id: "n1", sum: 0.14, value: 0.99, gradient: -0.005,
     symbolicEquation: "cos((x0*w1) + (x1*w3))", symbolicGradient: "y0'*(-sin....)"
   });
   results.push(result);

*/

// Write your code here....
`


const FlowChart = (props) => {
    // react state hooks
    const [elements, setElements] = useState(initialElements);
    const [elementsBackup, setElementsBackup] = useState(null);
    const [inputIdCounter, setInputIdCounter] = useState(initialElements.filter(element => element.id.includes("x")).length);
    const [nodeIdCounter, setNodeIdCounter] = useState(initialElements.filter(element => element.id.includes("n")).length);
    const [edgeIdCounter, setEdgeIdCounter] = useState(initialElements.filter(element => element.id.includes("w")).length);
    const [outputIdCounter, setOutputIdCounter] = useState(initialElements.filter(element => element.id.includes("y")).length);
    const [currentSelection, setCurrentSelection] = useState(null);
    const [learningRate, setLearningRate] = useState(0.5);
    const [loss, setLoss] = useState(0.0);
    const [lossHistory, setLossHistory] = useState([]);
    const [runBackpropagation, setRunBackpropagation] = useState(false);
    const [position, setPosition] = useState({ x: 0, y: 0, zoom: 1.0 });
    const [visualizationSettings, setVisualizationSettings] = useState("forwardpropagation");
    const [tutorialIndex, setTutorialIndex] = useState(null);
    const [customBackpropagationAlgorithm, setCustomBackpropagationAlgorithm] = useState(initialCustomBackpropagationAlgorithm);
    const [selectedBackpropagationAlgorithm, setSelectedBackpropagationAlgorithm] = useState("complete");
    const [editorMode, setEditorMode] = useState(false);
    const [editorConsole, setEditorConsole] = useState("");
    useEffect(() => {
        // helper function to clear values
        function clear() {
            for (let i in elements) {
                if (elements[i].id.includes("w") || elements[i].id.includes("x")) {
                    setNodeValue(elements[i].id, { symbolicEquation: NaN, symbolicSum: NaN, symbolicGradient: NaN, gradient: NaN });
                } else {
                    setNodeValue(elements[i].id, { symbolicEquation: NaN, symbolicSum: NaN, symbolicGradient: NaN, value: NaN, gradient: NaN });
                }
            }
        }
        try {
            // calculate weight ranges for coloring the edges
            let edges = elements.filter(element => element.source != undefined);
            let minEdgeValue = Math.min.apply(Math, edges.map(function (e) { return e.data.value; })) - 0.000001;
            let maxEdgeValue = Math.max.apply(Math, edges.map(function (e) { return e.data.value; })) + 0.000001;
            let minEdgeGradient = Math.min.apply(Math, edges.map(function (e) { return e.data.gradient; })) - 0.000001;
            let maxEdgeGradient = Math.max.apply(Math, edges.map(function (e) { return e.data.gradient; })) + 0.000001;
            // color the edges
            for (let i in edges) {
                let node = elements.filter(element => element.id == edges[i].source)[0];
                if (visualizationSettings == "forwardpropagation") {
                    // update edge colors based on value
                    let opacity = Math.floor((50 * ((edges[i].data.value - minEdgeValue) / (maxEdgeValue - minEdgeValue)) + 50) * 2.5);
                    if (isNaN(opacity)) opacity = 50;
                    setNodeValue(edges[i].id, { color: node.data.color + opacity.toString(16) });
                } else if (visualizationSettings == "backpropagation") {
                    let opacity = Math.floor((50 * ((edges[i].data.gradient - minEdgeGradient) / (maxEdgeGradient - minEdgeGradient)) + 50) * 2.5);
                    if (isNaN(opacity)) opacity = 50;
                    setNodeValue(edges[i].id, { color: node.data.color + opacity.toString(16) });
                }
            }
            // run the backpropagation algorithm
            let results = []
            let backpropagation = new Function("graph", "math", "results", "");
            if (selectedBackpropagationAlgorithm == "custom") {
                backpropagation = new Function("graph", "math", "results", customBackpropagationAlgorithm);
            } else if (selectedBackpropagationAlgorithm == "minimal") {
                backpropagation = new Function("graph", "math", "results", minimalBackpropagationAlgorithm);
            } else {
                backpropagation = new Function("graph", "math", "results", completeBackpropagationAlgorithm);
            }
            backpropagation(elements, math, results);
            // clear previous backpropagation results
            clear();
            // set graph values based on the results
            for (let i in results) {
                if (results[i].id != undefined) {
                    // update node values
                    setNodeValue(results[i].id, { sum: results[i].sum, value: results[i].value, symbolicEquation: results[i].symbolicEquation, gradient: results[i].gradient, symbolicGradient: results[i].symbolicGradient });
                } else if (results[i].loss != undefined) {
                    // update loss
                    setLoss(results[i].loss);
                }
            }
            // clear console
            setEditorConsole("");
        } catch (e) {
            // clear everything on error
            setEditorConsole(e.toString());
            clear();
        }
        // reset
        setRunBackpropagation(false);
    }, [runBackpropagation]);
    // reactflow events
    const onMove = (flowTransform) => { setPosition(flowTransform); };
    const onElementsRemove = (elementsToRemove) => {
        // remove element
        setElements((els) => removeElements(elementsToRemove, els))
        setCurrentSelection(null);
        // run backpropagation
        setRunBackpropagation(true);
    };
    const onConnect = (params) => {
        // add edge
        var edge = { ...params, id: 'w' + edgeIdCounter, label: 'w' + edgeIdCounter + ': 1.0', data: { value: 1.0 } };
        setElements((els) => addEdge(edge, els));
        setEdgeIdCounter(edgeIdCounter + 1);
        // run backpropagation
        setRunBackpropagation(true);
    };
    const onSelectionChange = (els) => { setCurrentSelection(els); };
    // reactflow helper functions
    const pad = (str, length, char = ' ') => str.padStart((str.length + length) / 2, char).padEnd(length, char);
    const getNodeValue = (nodeId) => { try { return elements.find(element => element.id == nodeId) } catch (e) { return null } };
    const setNodeValue = (nodeId, args) => {
        setElements((els) =>
            els.map((e) => {
                if (e.id === nodeId) {
                    let sum = (args.sum === undefined ? e.data.sum : args.sum);
                    let symbolicSum = (args.symbolicSum === undefined ? e.data.symbolicSum : args.symbolicSum);
                    let value = (args.value === undefined ? e.data.value : args.value);
                    let activation = (args.activation === undefined ? e.data.activation : args.activation);
                    let symbolicEquation = (args.symbolicEquation === undefined ? e.data.symbolicEquation : args.symbolicEquation);
                    let gradient = (args.gradient === undefined ? e.data.gradient : args.gradient);
                    let symbolicGradient = (args.symbolicGradient === undefined ? e.data.symbolicGradient : args.symbolicGradient);
                    let t = (args.true === undefined ? e.data.true : args.true);
                    let nodeLabel = "";
                    let weightLabel = "";
                    if (visualizationSettings == "forwardpropagation") {
                        let nodeLabelValue = pad(String(nodeId) + ":\u2800" + String(parseFloat(Number(value).toFixed(4))).replace(/-/g, '\u2015'), 16, '\u2800');
                        let nodeLabelTrue = pad(" t" + nodeId.substring(1) + ":\u2800" + String(parseFloat(Number(t).toFixed(4))).replace(/-/g, '\u2015'), 13, '\u2800');
                        let nodeLabelActivation = pad(" \u03B1(x)=" + String(activation), 16, '\u2800');
                        if (nodeId.includes("y")) {
                            nodeLabel = nodeLabelValue + nodeLabelTrue + nodeLabelActivation;
                        } else if (nodeId.includes("x")) {
                            nodeLabel = nodeLabelValue;
                        } else {
                            nodeLabel = nodeLabelValue + nodeLabelActivation;
                        }
                        weightLabel = String(nodeId) + ": " + parseFloat(Number(value).toFixed(4));
                    } else if (visualizationSettings == "backpropagation") {
                        let nodeLabelValue = pad(String(nodeId) + ":\u2800" + String(parseFloat(Number(value).toFixed(4))).replace(/-/g, '\u2015'), 16, '\u2800');
                        let nodeLabelTrue = pad(" t" + nodeId.substring(1) + ":\u2800" + String(parseFloat(Number(t).toFixed(4))).replace(/-/g, '\u2015'), 13, '\u2800');
                        let nodeLabelGradient = pad(" " + String(nodeId) + "':\u2800" + String(parseFloat(Number(gradient).toFixed(4))).replace(/-/g, '\u2015'), 16, '\u2800');
                        if (nodeId.includes("y")) {
                            nodeLabel = nodeLabelValue + nodeLabelTrue + nodeLabelGradient;
                        } else {
                            nodeLabel = nodeLabelValue + nodeLabelGradient;
                        }
                        weightLabel = String(nodeId) + "': " + parseFloat(Number(gradient).toFixed(4));
                    }
                    let color = (args.color === undefined ? e.data.color : args.color);
                    return {
                        ...e,
                        data: {
                            sum: sum, symbolicSum: symbolicSum, value: value, activation: activation, symbolicEquation: symbolicEquation, gradient: gradient, true: t, label: nodeLabel,
                            color: color, symbolicGradient: symbolicGradient
                        },
                        label: weightLabel,
                        style: { border: "2px solid " + color, stroke: color }
                    };
                }
                return e;
            })
        );
    };
    // other helper functions
    const randomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
    const tutorials = filterChildren(props.children, c => { return c.type.name && c.type.name.toLowerCase() === 'step'; });
    const hideTutorials = () => { setTutorialIndex(null); };
    function gradientdescent(graph, learningRate) {
        // run gradient descent on all edges
        let edges = graph.filter(element => element.source != null);
        for (let j in edges) {
            let edge = edges[j];
            let newWeight = edge.data.value - learningRate * edge.data.gradient;
            setNodeValue(edge.id, { value: newWeight });
        }
        // run backpropagation
        setRunBackpropagation(true);
    }

    return (
        <div className="providerflow">
            <ReactFlowProvider>
                {editorMode ?
                    <aside style={{ width: "50%", padding: "0" }}>
                        <div style={{ borderBottom: "1px solid", paddingBottom: "4px", paddingTop: "2px", overflow: "auto" }}>
                            <div style={{ fontSize: "14px", marginLeft: "20px", float: "left" }}>
                                Message: <div style={{ display: "inline-block", color: "red" }}>{editorConsole ? editorConsole : "-"}</div>
                            </div>
                            <div style={{ float: "right" }}>
                                <label style={{ fontSize: "14px", paddingLeft: "25px", paddingRight: "10px" }} className="container">Custom
                                    <input
                                        type="radio"
                                        checked={selectedBackpropagationAlgorithm == "custom"}
                                        value={"custom"}
                                        name={"custom"}
                                        onChange={() => {
                                            setSelectedBackpropagationAlgorithm("custom");
                                            setRunBackpropagation(true);
                                        }}
                                    />
                                    <span style={{ height: "20px", width: "20px" }} class="checkmark"></span>
                                </label>
                                <label style={{ fontSize: "14px", paddingLeft: "25px", paddingRight: "10px" }} className="container">Minimal
                                    <input
                                        type="radio"
                                        checked={selectedBackpropagationAlgorithm == "minimal"}
                                        value={"minimal"}
                                        name={"minimal"}
                                        onChange={() => {
                                            setSelectedBackpropagationAlgorithm("minimal");
                                            setRunBackpropagation(true);
                                        }}
                                    />
                                    <span style={{ height: "20px", width: "20px" }} class="checkmark"></span>
                                </label>
                                <label style={{ fontSize: "14px", paddingLeft: "25px", paddingRight: "20px" }} className="container">Complete
                                    <input
                                        type="radio"
                                        checked={selectedBackpropagationAlgorithm == "complete"}
                                        value={"complete"}
                                        name={"complete"}
                                        onChange={() => {
                                            setSelectedBackpropagationAlgorithm("complete");
                                            setRunBackpropagation(true);
                                        }}
                                    />
                                    <span style={{ height: "20px", width: "20px" }} class="checkmark"></span>
                                </label>
                                <button className="button" style={{ float: "right", fontSize: "12px", margin: "0", padding: "2px 4px", borderStyle: "solid", borderWidth: "1px" }} type="button" onClick={() => { setEditorMode(!editorMode); }}>Hide</button>
                            </div>
                        </div>
                        <div style={{ overflowY: "scroll", height: "97vh" }}>
                            <Editor
                                value={
                                    selectedBackpropagationAlgorithm == "complete" ? completeBackpropagationAlgorithm : (selectedBackpropagationAlgorithm == "minimal" ? minimalBackpropagationAlgorithm : customBackpropagationAlgorithm)
                                }
                                onValueChange={code => {
                                    if (selectedBackpropagationAlgorithm == "custom") {
                                        // read only
                                        setCustomBackpropagationAlgorithm(code);
                                        setRunBackpropagation(true);
                                    }
                                }}
                                highlight={code => Prism.highlight(code, Prism.languages.javascript, 'javascript')}
                                padding={10}
                                tabSize={2}
                                style={{
                                    fontFamily: '"Fira code", "Fira Mono", monospace',
                                    fontSize: 12,
                                }}
                            />
                        </div>
                    </aside>
                    :
                    <aside>
                        <button className="button" style={{ float: "right", fontSize: "12px", margin: "0", padding: "2px 4px", borderStyle: "solid", borderWidth: "1px" }} type="button" onClick={() => { setEditorMode(!editorMode); }}>Editor</button>
                        <div>
                            <p></p>
                            <div className="grid-title">Architecture</div>
                            <p></p>
                            <div className="centered">
                                <button style={{ width: "45%" }} type="button" className="button" onClick={() => {
                                    let newElement = {
                                        id: 'x' + inputIdCounter, data: { label: 'x' + inputIdCounter + ": 0.0", value: 0.0, activation: 'x', color: '#4A90E2' },
                                        position: { x: randomInt(20, 100) - position.x, y: randomInt(10, 50) - position.y }, sourcePosition: 'right', type: 'input',
                                        style: { border: "2px solid #4A90E2", stroke: '#4A90E2' }
                                    };
                                    setElements([...elements, newElement]);
                                    setInputIdCounter(inputIdCounter + 1);
                                    setCurrentSelection([{ id: 'x' + inputIdCounter }]);
                                    setLossHistory([]);
                                }}>Add Input</button>
                                <button style={{ width: "45%" }} type="button" className="button" onClick={() => {
                                    let nodeLabelValue = pad('y' + outputIdCounter + ":\u28000.0", 16, '\u2800');
                                    let nodeLabelTrue = pad(" t" + outputIdCounter + ":\u28000.0", 13, '\u2800');
                                    let nodeLabelActivation = pad(" \u03B1(x)=x", 16, '\u2800');
                                    let newElement = {
                                        id: 'y' + outputIdCounter, data: { label: nodeLabelValue + nodeLabelTrue + nodeLabelActivation, value: 0.0, activation: 'x', true: 0.0, color: '#D0021B', },
                                        position: { x: randomInt(400, 500) - position.x, y: randomInt(10, 50) - position.y }, targetPosition: 'left', type: 'output',
                                        style: { border: "2px solid #D0021B", stroke: '#D0021B' }
                                    };
                                    setElements([...elements, newElement]);
                                    setOutputIdCounter(outputIdCounter + 1);
                                    setCurrentSelection([{ id: 'y' + outputIdCounter }]);
                                    setLossHistory([]);
                                }}>Add Output</button>
                            </div>
                            <div className="centered">
                                <button style={{ width: "45%" }} type="button" className="button" onClick={() => {
                                    let nodeLabelValue = pad('n' + nodeIdCounter + ":\u28000.0", 16, '\u2800');
                                    let nodeLabelActivation = pad(" \u03B1(x)=x", 16, '\u2800');
                                    let newElement = {
                                        id: 'n' + nodeIdCounter, data: { label: nodeLabelValue + nodeLabelActivation, value: 0.0, activation: 'x', color: '#4A4A4A' },
                                        position: { x: randomInt(200, 300) - position.x, y: randomInt(10, 50) - position.y }, targetPosition: 'left', sourcePosition: 'right',
                                        style: { border: "2px solid #4A4A4A", stroke: '#4A4A4A' }
                                    };
                                    setElements([...elements, newElement]);
                                    setNodeIdCounter(nodeIdCounter + 1);
                                    setCurrentSelection([{ id: 'n' + nodeIdCounter }]);
                                    setLossHistory([]);
                                }}>Add Node</button>
                                <button style={{ width: "45%" }} type="button" className="button" onClick={() => {
                                    // clear everything
                                    setElements([]);
                                    setInputIdCounter(0);
                                    setNodeIdCounter(0);
                                    setEdgeIdCounter(0);
                                    setOutputIdCounter(0);
                                    setCurrentSelection(null);
                                    setLossHistory([]);
                                    // run backpropagation
                                    setRunBackpropagation(true);
                                }}>Clear All</button>
                            </div>
                            <div>
                                <p></p>
                                <div className="grid-title">Node Parameters</div>
                                <p></p>
                                <table>
                                    <tr>
                                        <th>Selected:</th>
                                        {currentSelection != null ? <th>{currentSelection[0].id}</th> : <th>Please select a node or weight!</th>}
                                    </tr>
                                    {currentSelection != null && (currentSelection[0].id.includes("x") || currentSelection[0].id.includes("w")) ?
                                        <tr><th>Value:</th><th><input type="text" value={getNodeValue(currentSelection[0].id).data.value} onChange={(e) => {
                                            // update node or edge value
                                            setNodeValue(currentSelection[0].id, { value: e.target.value });
                                            // run backpropagation
                                            setRunBackpropagation(true);
                                        }} /></th></tr>
                                        : ""}
                                    {currentSelection != null && !currentSelection[0].id.includes("w") && !currentSelection[0].id.includes("x") ?
                                        <tr><th>Activation:</th><th><input type="text" value={getNodeValue(currentSelection[0].id).data.activation} onChange={(e) => {
                                            // update activation function
                                            setNodeValue(currentSelection[0].id, { activation: e.target.value });
                                            // run backpropagation
                                            setRunBackpropagation(true);
                                        }} /></th></tr>
                                        : ""}
                                    {currentSelection != null && currentSelection[0].id.includes("y") ?
                                        <tr><th>True:</th><th><input type="text" value={getNodeValue(currentSelection[0].id).data.true} onChange={(e) => {
                                            // update activation function
                                            setNodeValue(currentSelection[0].id, { true: e.target.value });
                                            // run backpropagation
                                            setRunBackpropagation(true);
                                        }} /></th></tr>
                                        : ""}
                                    {currentSelection != null && !currentSelection[0].id.includes("w") ?
                                        <tr><th>Color:</th><th>
                                            <GithubPicker className="centered" colors={['#D0021B', '#F5A623', '#F8E71C', '#8B572A', '#7ED321', '#417505', '#BD10E0',
                                                '#9013FE', '#4A90E2', '#50E3C2', '#B8E986', '#000000', '#4A4A4A', '#9B9B9B']} triangle='hide'
                                                onChangeComplete={(color) => {
                                                    // update node color
                                                    setNodeValue(currentSelection[0].id, { color: color.hex });
                                                    // update color of outgoing edges/weights
                                                    let edges = elements.filter(element => element.source == currentSelection[0].id,);
                                                    for (let j in edges) {
                                                        let edge = edges[j];
                                                        setNodeValue(edge.id, { color: color.hex });
                                                    }
                                                    // run backpropagation
                                                    setRunBackpropagation(true);
                                                }} />
                                        </th></tr>
                                        : ""}
                                </table>
                            </div>
                            <div>
                                <p></p>
                                <div className="grid-title">Gradient Descent</div>
                                <p></p>
                                <MathJax.Context input='ascii'>
                                    <div>
                                        Loss: <MathJax.Node inline>{'sum_i 1/2 (t_i-y_i)^2 = ' + parseFloat(Number(loss).toFixed(4))}</MathJax.Node>
                                    </div>
                                </MathJax.Context>
                        Learning Rate: <input style={{ width: "80px" }} type="text" value={learningRate} onChange={(e) => { setLearningRate(e.target.value) }} />
                                <button type="button" className="button" onClick={() => {
                                    if (elementsBackup == null) setElementsBackup(elements);
                                    setLossHistory([...lossHistory, loss]);
                                    gradientdescent(elements, learningRate);
                                }}>Train</button>
                                <button type="button" className="button" onClick={() => {
                                    if (elementsBackup != null) {
                                        setElements(elementsBackup);
                                        setElementsBackup(null);
                                    }
                                    setLossHistory([]);
                                    // run backpropagation
                                    setRunBackpropagation(true);
                                }}>Reset</button>
                                {!isNaN(loss) ?
                                    <VictoryChart domain={{ x: [0, Math.max(10, lossHistory.length)], y: [0, Math.max(Math.max(Math.max.apply(Math, lossHistory), 0), loss)] }}>
                                        <VictoryAxis label="iterations" />
                                        <VictoryAxis dependentAxis />
                                        <VictoryScatter data={[{ x: lossHistory.length, y: loss }]} size={5} style={{ data: { fill: "#c43a31" } }} />
                                        {lossHistory.map((item, index) => {
                                            return <VictoryScatter data={[{ x: index, y: item }]} size={5} style={{ data: { fill: "#c43a31" } }} />
                                        })}
                                        <VictoryLine style={{ data: { stroke: "#c43a31" } }} data={[{ x: lossHistory.length - 1, y: lossHistory[lossHistory.length - 1] }, { x: lossHistory.length, y: loss },]} />
                                        {lossHistory.map((item, index, array) => {
                                            if (index > 0) {
                                                return <VictoryLine style={{ data: { stroke: "#c43a31" } }} data={[
                                                    { x: index - 1, y: array[index - 1] },
                                                    { x: index, y: array[index] },
                                                ]} />
                                            }
                                        })}
                                    </VictoryChart>
                                    : false}
                            </div>
                            <div>
                                <p></p>
                                <div className="grid-title">Visualization Settings</div>
                                <p></p>
                                <label style={{ fontSize: "12px" }} className="container">Forwardpropagation (Weights)
                                    <input
                                        type="radio"
                                        checked={visualizationSettings == "forwardpropagation"}
                                        value={"forwardpropagation"}
                                        name={"forwardpropagation"}
                                        onChange={() => {
                                            setVisualizationSettings("forwardpropagation");
                                            // run backpropagation
                                            setRunBackpropagation(true);
                                        }}
                                    />
                                    <span style={{ height: "20px", width: "20px" }} class="checkmark"></span>
                                </label>
                                <p></p>
                                <label style={{ fontSize: "12px" }} className="container">Backpropagation (Gradients)
                                    <input
                                        type="radio"
                                        checked={visualizationSettings == "backpropagation"}
                                        value={"backpropagation"}
                                        name={"backpropagation"}
                                        onChange={() => {
                                            setVisualizationSettings("backpropagation");
                                            // run backpropagation
                                            setRunBackpropagation(true);
                                        }}
                                    />
                                    <span style={{ height: "20px", width: "20px" }} class="checkmark"></span>
                                </label>
                            </div>
                            <div>
                                <p></p>
                                <div className="grid-title">Tutorial</div>
                                <p></p>
                                {props.tutorials.map((item, index) => (
                                    <a href="#" onClick={() => { setTutorialIndex(index) }}><h3 style={{ margin: "10px" }}>{item}</h3></a>
                                ))}
                                <p></p>
                                <p></p>
                                <p>(This tutorial was created using <a href="https://idyll-lang.org">Idyll)</a></p>
                            </div>
                        </div>
                    </aside>
                }
                <div className="reactflow-wrapper">
                    <p></p>
                    {visualizationSettings == "forwardpropagation" ?
                        <div className="centered" dangerouslySetInnerHTML={{
                            __html: (currentSelection != null && !currentSelection[0].id.includes("w"))
                                ? "<h2>" + currentSelection[0].id + " = " + getNodeValue(currentSelection[0].id).data.symbolicEquation + " = " + parseFloat(Number(getNodeValue(currentSelection[0].id).data.value).toFixed(4)) + "</h2>"
                                : "<h2>Please select a node!</h2>"
                        }} />
                        :
                        <div className="centered" dangerouslySetInnerHTML={{
                            __html: (currentSelection != null)
                                ? "<h2>" + currentSelection[0].id + "' = " + getNodeValue(currentSelection[0].id).data.symbolicGradient + " = " + parseFloat(Number(getNodeValue(currentSelection[0].id).data.gradient).toFixed(4)) + "</h2>"
                                : "<h2>Please select a node!</h2>"
                        }} />}
                    <p></p>
                    <ReactFlow
                        elements={elements}
                        onElementsRemove={onElementsRemove}
                        onConnect={onConnect}
                        onSelectionChange={onSelectionChange}
                        onMove={onMove}
                    />
                </div>
            </ReactFlowProvider>
            {tutorialIndex != null ? <AriaModal
                titleText="demo one"
                initialFocus="#demo-one-modal"
                onExit={hideTutorials}
                underlayStyle={{ paddingTop: '2em' }}
            >
                <div id="demo-one-modal" className="modal">
                    <div className="modal-body">
                        {tutorials[tutorialIndex]}
                    </div>
                </div>
            </AriaModal>
                : false}
        </div>
    );
}

module.exports = FlowChart;