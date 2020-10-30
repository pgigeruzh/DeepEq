/*
    ToDO: 
    - Convert to dynamic programming
    - Node connection validation 1:n
    - Bug when backpropagating multiple connection neurons (due to summation of errors)
*/

import React, { useState, useEffect } from 'react';
import ReactFlow, { ReactFlowProvider, removeElements, addEdge, Controls } from 'react-flow-renderer';
import MathJax from 'react-mathjax2';
import { GithubPicker, SketchPicker, TwitterPicker } from 'react-color'
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

const FlowChart = () => {
    // react state hooks
    const [elements, setElements] = useState(initialElements);
    const [inputIdCounter, setInputIdCounter] = useState(initialElements.filter(element => element.id.includes("x")).length);
    const [nodeIdCounter, setNodeIdCounter] = useState(initialElements.filter(element => element.id.includes("n")).length);
    const [edgeIdCounter, setEdgeIdCounter] = useState(initialElements.filter(element => element.id.includes("w")).length);
    const [outputIdCounter, setOutputIdCounter] = useState(initialElements.filter(element => element.id.includes("y")).length);
    const [currentSelection, setCurrentSelection] = useState(null);
    const [learningRate, setLearningRate] = useState(0.5);
    const [loss, setLoss] = useState(0.0);
    const [runBackpropagation, setRunBackpropagation] = useState(false);
    const [position, setPosition] = useState({ x: 0, y: 0, zoom: 1.0 });
    const [visualizationSettings, setVisualizationSettings] = useState("forwardpropagation");
    useEffect(() => {
        // run forward propagation for each initial node
        let forwardresults = [];
        let edges = elements.filter(element => element.source != undefined);
        let minEdgeValue = Math.min.apply(Math, edges.map(function (e) { return e.data.value; })) - 0.000001;
        let maxEdgeValue = Math.max.apply(Math, edges.map(function (e) { return e.data.value; })) + 0.000001;
        let minEdgeGradient = Math.min.apply(Math, edges.map(function (e) { return e.data.gradient; })) - 0.000001;
        let maxEdgeGradient = Math.max.apply(Math, edges.map(function (e) { return e.data.gradient; })) + 0.000001;
        let initialNodes = elements.filter(element => element.id.includes("y"));
        for (let i in initialNodes) { forwardpropagation(elements, initialNodes[i], forwardresults); }
        for (let i in forwardresults) {
            // update node values
            setNodeValue(forwardresults[i].id, { sum: forwardresults[i].sum, value: forwardresults[i].value, symbolicEquation: forwardresults[i].symbolicEquation });
            // update edge color based on weight
            if (visualizationSettings == "forwardpropagation") {
                let outgoingEdges = elements.filter(element => element.source == forwardresults[i].id);
                for (let j in outgoingEdges) {
                    let opacity = Math.floor((50 * ((outgoingEdges[j].data.value - minEdgeValue) / (maxEdgeValue - minEdgeValue)) + 50) * 2.5);
                    setNodeValue(outgoingEdges[j].id, { color: forwardresults[i].color + opacity.toString(16) });
                }
            }
        }
        // update loss
        var loss = 0.0;
        for (let i in initialNodes) {
            let nodes = forwardresults.filter(element => element.id.includes(initialNodes[i].id));
            if (nodes.length > 0) {
                let t = initialNodes[i].data.true;
                let y = nodes[0].value;
                loss += 0.5 * (t - y) ** 2;
            }
        }
        setLoss(loss);
        // run backpropagation
        let backpropresults = [];
        initialNodes = elements.filter(element => element.id.includes("x"));
        for (let i in initialNodes) { backpropagation(elements, forwardresults, initialNodes[i], backpropresults); }
        for (let i in backpropresults) {
            // update gradients
            setNodeValue(backpropresults[i].id, { gradient: backpropresults[i].gradient, symbolicGradient: backpropresults[i].symbolicGradient });
            // update edge colors based on gradients
            if (visualizationSettings == "backpropagation" && backpropresults[i].color != undefined) {
                let opacity = Math.floor((50 * ((backpropresults[i].gradient - minEdgeGradient) / (maxEdgeGradient - minEdgeGradient)) + 50) * 2.5);
                setNodeValue(backpropresults[i].id, { color: backpropresults[i].color + opacity.toString(16) });
            }
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
    // helper functions
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
    const randomInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
    // main algorithms
    function forwardpropagation(graph, node, result) {
        // get all ingoing edges from node
        let edges = graph.filter(element => element.target == node.id);
        if (edges.length > 0) {
            try {
                // calculate weighted sum of perceptron e.g. x1*w1+...+xn*wn
                let sum = 0;
                let symbolicEquation = "";
                let symbolicSum = "";
                for (let j in edges) {
                    let edge = edges[j];
                    let nextNode = graph.filter(element => element.id == edge.source)[0];
                    // recursive call
                    // ToDo: convert to dynamic programming
                    let [nextNodeValue, nextNodeSymbolicEquation] = forwardpropagation(graph, nextNode, result);
                    // calculate necessary values
                    sum += edge.data.value * nextNodeValue;
                    symbolicEquation += "<div style='color:" + edge.data.color + ";display:inline;'>(" + nextNodeSymbolicEquation + "*" + edge.id + ")</div>";
                    symbolicSum += "<div style='color:" + edge.data.color + ";display:inline;'>(" + nextNode.id + "*" + edge.id + ")</div>";
                    if (j < edges.length - 1) { symbolicEquation += "+"; symbolicSum += "+" };
                }
                // calculate activation function value
                let value = math.evaluate(node.data.activation.replace('x', sum));
                // calculate summation of symbolic equation
                let activation = node.data.activation;
                if ((activation[activation.indexOf('x') - 1] != '(') || (activation[activation.indexOf('x') + 1] != ')')) {
                    activation = activation.replace('x', "(x)")
                }
                symbolicEquation = "<div style='color:" + node.data.color + ";display:inline;'>" + activation.replace('x', symbolicEquation) + "</div>";
                // assignn result
                result.push({ id: node.id, sum: sum, symbolicSum: symbolicSum, value: value, symbolicEquation: symbolicEquation, color: node.data.color });
                return [value, symbolicEquation];
            } catch (e) {
                return e;
            }
        } else if (node.id.includes("x")) {
            // input node
            result.push({ id: node.id, value: node.data.value, symbolicEquation: node.id, color: node.data.color });
            return [node.data.value, node.id];
        } else {
            // node without connections
            return NaN;
        }
    }
    function backpropagation(graph, forwardresults, node, result) {
        // get all outgoing edges from node
        let edges = graph.filter(element => element.source == node.id);
        if (edges.length > 0) {
            try {
                let totalGradient = 0.0;
                let totalSymbolicGradient = "";
                for (let j in edges) {
                    let edge = edges[j];
                    let nextNode = graph.filter(element => element.id == edge.target)[0];
                    // recursive call
                    // ToDo: convert to dynamic programming
                    let [nextNodeGradient, nextSymbolicGradient] = backpropagation(graph, forwardresults, nextNode, result);
                    // use latest data from forward propagation
                    let tmp = forwardresults.filter(element => element.id == node.id)[0];
                    let nodeColor = tmp.color;
                    let nodeSymbolicSum = tmp.symbolicSum;
                    let nodeSum = tmp.sum;
                    let nodeValue = tmp.value;
                    // calculate gradient
                    let outerDerivative = math.derivative(node.data.activation, 'x');
                    totalGradient += nextNodeGradient * outerDerivative.evaluate({ x: nodeSum }) * edge.data.value;
                    totalSymbolicGradient += "<div style='color:" + nextNode.data.color + ";display:inline;'>" + nextSymbolicGradient + "</div>" + "*("
                        + "<div style='color:" + node.data.color + ";display:inline;'>" + outerDerivative.toString().replace(/\s+/g, '').replace('x', nodeSymbolicSum) + "</div>" + "*"
                        + "<div style='color:" + node.data.color + ";display:inline;'>" + edge.id + "</div>" + ")";
                    if (j < edges.length - 1) totalSymbolicGradient += "+";
                    // assign result
                    let edgeGradient = {
                        id: edge.id,
                        gradient: nextNodeGradient * nodeValue,
                        symbolicGradient: "<div style='color:" + nextNode.data.color + ";display:inline;'>" + nextSymbolicGradient + "</div>" + "*"
                            + "<div style='color:" + node.data.color + ";display:inline;'>" + node.id + "</div>",
                        color: nodeColor
                    };
                    result.push(edgeGradient);
                }
                let nodeGradient = { id: node.id, gradient: totalGradient, symbolicGradient: totalSymbolicGradient };
                result.push(nodeGradient);
                return [totalGradient, node.id + "'"];
            } catch (e) {
                return e;
            }
        } else if (node.id.includes("y")) {
            // output node

            // use latest data from forward propagation
            let tmp = forwardresults.filter(element => element.id == node.id)[0];
            let nodeSum = tmp.sum;
            let nodeSymbolicSum = tmp.symbolicSum;
            let nodeValue = tmp.value;
            // calculate derivative of activation function
            let outerDerivative = math.derivative(node.data.activation, 'x');
            // assign results
            let nodeGradient = {
                id: node.id,
                gradient: -1 * (node.data.true - nodeValue) * outerDerivative.evaluate({ x: nodeSum }),
                symbolicGradient: "<div style='color:" + node.data.color + ";display:inline;'>" + "-1*(t" + node.id.substring(1) + "-" + node.id + ")*"
                    + outerDerivative.toString().replace(/\s+/g, '').replace('x', nodeSymbolicSum) + "</div>"
            };
            result.push(nodeGradient);
            return [nodeGradient.gradient, node.id + "'"];
        } else {
            // node without connections
            return NaN;
        }
    }
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
                <aside>
                    <div className="grid-title">Architecture</div>
                    <p></p>
                    <div className="centered">
                        <button style={{ width: "45%" }} type="button" className="button" onClick={() => {
                            let newElement = {
                                id: 'x' + inputIdCounter, data: { label: 'x' + inputIdCounter + ": 0.0" + ", \u0278(x)=x", value: 0.0, activation: 'x', color: '#4A90E2' },
                                position: { x: randomInt(20, 100) - position.x, y: randomInt(10, 50) - position.y }, sourcePosition: 'right', type: 'input',
                                style: { border: "2px solid #4A90E2", stroke: '#4A90E2' }
                            };
                            setElements([...elements, newElement]);
                            setInputIdCounter(inputIdCounter + 1);
                            setCurrentSelection([{ id: 'x' + inputIdCounter }]);
                        }}>Add Input</button>
                        <button style={{ width: "45%" }} type="button" className="button" onClick={() => {
                            let newElement = {
                                id: 'y' + outputIdCounter, data: { label: 'y' + outputIdCounter + ": 0.0" + ", \u0278(x)=x", value: 0.0, activation: 'x', true: 0.0, color: '#D0021B', },
                                position: { x: randomInt(400, 500) - position.x, y: randomInt(10, 50) - position.y }, targetPosition: 'left', type: 'output',
                                style: { border: "2px solid #D0021B", stroke: '#D0021B' }
                            };
                            setElements([...elements, newElement]);
                            setOutputIdCounter(outputIdCounter + 1);
                            setCurrentSelection([{ id: 'y' + outputIdCounter }]);
                        }}>Add Output</button>
                    </div>
                    <div className="centered">
                        <button style={{ width: "45%" }} type="button" className="button" onClick={() => {
                            let newElement = {
                                id: 'n' + nodeIdCounter, data: { label: 'n' + nodeIdCounter + ": 0.0" + ", \u0278(x)=x", value: 0.0, activation: 'x', color: '#4A4A4A' },
                                position: { x: randomInt(200, 300) - position.x, y: randomInt(10, 50) - position.y }, targetPosition: 'left', sourcePosition: 'right',
                                style: { border: "2px solid #4A4A4A", stroke: '#4A4A4A' }
                            };
                            setElements([...elements, newElement]);
                            setNodeIdCounter(nodeIdCounter + 1);
                            setCurrentSelection([{ id: 'n' + nodeIdCounter }]);
                        }}>Add Node</button>
                        <button style={{ width: "45%" }} type="button" className="button" onClick={() => {
                            setElements([]);
                            setInputIdCounter(0);
                            setNodeIdCounter(0);
                            setEdgeIdCounter(0);
                            setOutputIdCounter(0);
                            setCurrentSelection(null);
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
                        Learning Rate: <input style={{ width: "130px" }} type="text" value={learningRate} onChange={(e) => { setLearningRate(e.target.value) }} />
                        <button type="button" className="button" onClick={() => {
                            gradientdescent(elements, learningRate);
                        }}>Train</button>
                    </div>
                    <div>
                        <p></p>
                        <div className="grid-title">Visualization Settings</div>
                        <p></p>
                        <input type="radio" id="forwardpropagation" name="visualization" value="forwardpropagation" checked={visualizationSettings == "forwardpropagation"} onClick={() => {
                            setVisualizationSettings("forwardpropagation");
                            // run backpropagation
                            setRunBackpropagation(true);
                        }} />
                        <label for="forwardpropagation">Forwardpropagation (Weights)</label>
                        <br></br>
                        <input type="radio" id="backpropagation" name="visualization" value="backpropagation" checked={visualizationSettings == "backpropagation"} onClick={() => {
                            setVisualizationSettings("backpropagation");
                            // run backpropagation
                            setRunBackpropagation(true);
                        }} />
                        <label for="backpropagation">Backpropagation (Gradients)</label>
                    </div>
                </aside>
                <div className="reactflow-wrapper">
                    <p></p>
                    {visualizationSettings == "forwardpropagation" ?
                        <div className="centered" dangerouslySetInnerHTML={{
                            __html: (currentSelection != null && !currentSelection[0].id.includes("w"))
                                ? currentSelection[0].id + " = " + getNodeValue(currentSelection[0].id).data.symbolicEquation + " = " + parseFloat(Number(getNodeValue(currentSelection[0].id).data.value).toFixed(4))
                                : "Please select a node!"
                        }} />
                        :
                        <div className="centered" dangerouslySetInnerHTML={{
                            __html: (currentSelection != null)
                                ? currentSelection[0].id + "' = " + getNodeValue(currentSelection[0].id).data.symbolicGradient + " = " + parseFloat(Number(getNodeValue(currentSelection[0].id).data.gradient).toFixed(4))
                                : "Please select a node!"
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
        </div>
    );
}

module.exports = FlowChart;