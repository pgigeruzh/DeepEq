import MathJax from 'react-mathjax2';
const React = require('react');
const math = require('mathjs');
import { VictoryChart, VictoryAxis, VictoryLine, VictoryScatter } from 'victory';

class ExerciseCreator extends React.PureComponent {

    static defaultProps = {
        // restrict max. number of layers
        maxNumberOfLayers: 7,
    }

    constructor(props) {
        super(props);
        this.state = {
            // input
            x: 0.5,
            // true label
            t: 0.9,
            // weights
            weights: [8.0, ""],
            // weight history for gradient descent
            weightsHistory: [],
            // activation functions
            activations: ['x', 'sin(x)'],
            // loss function
            loss: "1/2 (t - y)^2",
            // loss history for gradient descent
            lossHistory: [],
            // history of all predictions
            predictionHistory: [],
            // learning rate
            learningRate: 1.5,
            // visualization radio button value
            radio: "loss",
        }
    }

    generateFormula() {
        // generate symbolic prediction formula e.g. y = sin(xw)
        var formula = 'x';
        for (var i = 0; i < this.state.weights.length; i++) {
            var activation = this.state.activations[i];
            var weight = this.state.weights[i];
            if (weight) {
                formula = '(' + activation.replace('x', formula) + ')' + '*w_' + i;
            } else {
                formula = '(' + activation.replace('x', formula) + ')';
            }
        }
        return (math.parse(formula).toString({ parenthesis: 'auto' }));
    }

    render() {
        try {
            // calculate formula
            var formulaSymbolicEquation = this.generateFormula();
            var formulaNumericEquation = formulaSymbolicEquation.replace('x', this.state.x);
            for (var i = 0; i < this.state.weights.length; i++) { formulaNumericEquation = formulaNumericEquation.replace('w_' + i, parseFloat(Number(this.state.weights[i]).toFixed(4))); }
            var formulaNumericSolution = math.evaluate(formulaNumericEquation);
            if (isNaN(formulaNumericSolution)) throw "error";
            // calculate loss
            var lossSymbolicEquation = this.state.loss;
            var lossNumericEquation = this.state.loss.replace('t', this.state.t).replace('y', parseFloat(Number(formulaNumericSolution).toFixed(4)));
            var lossNumericSolution = math.evaluate(lossNumericEquation);
            if (isNaN(lossNumericSolution)) throw "error";
            // calculate derivatives
            var partialLossSymbolicEquations = [];
            var partialLossSymbolicDerivatives = [];
            var partialLossNumericDerivativeSolutions = [];
            for (var i = 0; i < this.state.weights.length; i++) {
                if (this.state.weights[i]) {
                    var equation = lossSymbolicEquation.replace('y', formulaSymbolicEquation).replace('x', this.state.x).replace('t', this.state.t);
                    for (var j = 0; j < this.state.weights.length; j++) {
                        if (i != j) {
                            equation = equation.replace('w_' + j, this.state.weights[j]);
                        }
                    }
                    partialLossSymbolicEquations.push(equation);
                    var derivative = math.derivative(equation, 'w_' + i, { simplify: false }).toString({ parenthesis: 'auto' });
                    partialLossSymbolicDerivatives.push(derivative);
                    var derivativeSolution = math.evaluate(derivative.replace(new RegExp('w_' + i, 'g'), this.state.weights[i]));
                    if (isNaN(lossNumericSolution)) throw "error";
                    partialLossNumericDerivativeSolutions.push(derivativeSolution);
                } else {
                    partialLossSymbolicEquations.push("");
                    partialLossSymbolicDerivatives.push("");
                    partialLossNumericDerivativeSolutions.push(0.0);
                }
            }
        } catch (e) {
            // default values e.g. when equation is not valid
            var formulaSymbolicEquation = "NaN";
            var formulaNumericEquation = "NaN"
            var formulaNumericSolution = 0;
            var lossSymbolicEquation = "NaN";
            var lossNumericEquation = "NaN";
            var lossNumericSolution = 0;
            var partialLossSymbolicEquations = [];
            var partialLossSymbolicDerivatives = [];
            var partialLossNumericDerivativeSolutions = [];
        }
        return (
            <div className="exercise-creator">
                <p className="title">Deep Learning Theory for High School Students</p>
                <div style={{margin: '40px'}}><div className="centered"><a className="subtitle" href="#idyll-scroll-0">Show Tutorial</a></div></div>
                <div className="grid-container">
                    <div className="grid-item">
                        <div className="grid-title">Architecture</div>
                        <MathJax.Context input='ascii'>
                            <div style={{ textAlign: 'right' }}>
                                Input <MathJax.Node inline>{'x'}</MathJax.Node>:&nbsp;
                                    <input type="text" style={{ width: "100px" }} value={this.state.x} onChange={(e) => { this.setState({ x: e.target.value }); }} />
                            </div>
                        </MathJax.Context>
                        <table>
                            <tr>
                                <th>Layer</th>
                                <th style={{ width: "100%" }}>Activation</th>
                                <th>Weight</th>
                            </tr>
                            {this.state.weights.map((item, index) => (
                                <MathJax.Context input='ascii'>
                                    <tr>
                                        <th>{index}</th>
                                        <th>
                                            <input type="text" style={{ width: "100%" }} value={this.state.activations[index]} onChange={(e) => {
                                                var activations = [...this.state.activations];
                                                activations[index] = e.target.value;
                                                this.setState({ activations: activations });
                                            }} />
                                        </th>
                                        <th>
                                            <MathJax.Node inline>{'w_' + index}</MathJax.Node>:&nbsp;
                                                <input type="text" style={{ width: "100px" }} value={item} onChange={(e) => {
                                                var weights = [...this.state.weights];
                                                weights[index] = e.target.value;
                                                this.setState({ weights: weights });
                                            }} />
                                        </th>
                                    </tr>
                                </MathJax.Context>
                            ))}
                        </table>
                        <div style={{ textAlign: 'right' }}>
                            <button type="button" className="button" onClick={() => {
                                // add layer
                                if (this.state.weights.length < this.props.maxNumberOfLayers - 1) {
                                    this.setState({
                                        weights: [...this.state.weights, 1.0],
                                        activations: [...this.state.activations, 'sin(x)'],
                                    });
                                }
                            }}>Add Layer</button>
                            <button type="button" className="button" onClick={() => {
                                // remove layer
                                if (this.state.weights.length > 1) {
                                    this.setState({
                                        weights: [...this.state.weights].slice(0, -1),
                                        activations: [...this.state.activations].slice(0, -1),
                                    });
                                }
                            }}>Remove Layer</button>
                        </div>
                        <div>
                            Symbolic Equation:
                            <MathJax.Context input='ascii'>
                                <div style={{ overflowX: 'scroll', overflowY: 'hide', paddingBottom: '5px' }}>
                                    <MathJax.Node>{"y=" + formulaSymbolicEquation}</MathJax.Node>
                                </div>
                            </MathJax.Context>
                            Numeric Solution:
                            <MathJax.Context input='ascii'>
                                <div style={{ overflowX: 'scroll', overflowY: 'hide', paddingBottom: '5px' }}>
                                    <MathJax.Node>{"y=" + formulaNumericEquation + "=" + parseFloat(formulaNumericSolution.toFixed(4))}</MathJax.Node>
                                </div>
                            </MathJax.Context>
                        </div>
                    </div>
                    <div className="grid-item">
                        <div className="grid-title">Loss Function</div>
                        <MathJax.Context input='ascii'>
                            <div>
                                Loss: <input type="text" style={{ width: "120px" }} value={this.state.loss} onChange={(e) => { this.setState({ loss: e.target.value }); }} />
                            True <MathJax.Node>{"t"}</MathJax.Node>: <input type="text" style={{ width: "100px" }} value={this.state.t} onChange={(e) => { this.setState({ t: e.target.value }); }} />
                            </div>
                        </MathJax.Context>
                        Symbolic Equation:
                        <MathJax.Context input='ascii'>
                            <div style={{ overflowX: 'scroll', overflowY: 'hide', paddingBottom: '5px' }}>
                                <MathJax.Node>{"loss=" + lossSymbolicEquation + "=" + lossSymbolicEquation.replace('y', formulaSymbolicEquation)}</MathJax.Node>
                            </div>
                        </MathJax.Context>
                        Numeric Solution:
                        <MathJax.Context input='ascii'>
                            <div style={{ overflowX: 'scroll', overflowY: 'hide', paddingBottom: '5px' }}>
                                <MathJax.Node>{"loss=" + lossNumericEquation + "=" + parseFloat(lossNumericSolution.toFixed(4))}</MathJax.Node>
                            </div>
                        </MathJax.Context>
                        <div className="grid-title">Gradient Descent</div>
                        Learning Rate: <input type="text" style={{ width: "60px" }} value={this.state.learningRate} onChange={(e) => { this.setState({ learningRate: e.target.value }); }} />
                        <button type="button" className="button" onClick={() => {
                            // save old prediction
                            var predictionHistory = [...this.state.predictionHistory, formulaNumericSolution];
                            // save old loss
                            var lossHistory = [...this.state.lossHistory, lossNumericSolution];
                            // save old weights
                            var weightsHistory = [...this.state.weightsHistory, this.state.weights];
                            // run gradient descent: w_(new) = w_(old) - learning_rate * loss'
                            var newWeights = partialLossNumericDerivativeSolutions.map((item, index) => this.state.weights[index] != "" ? (this.state.weights[index] - this.state.learningRate * partialLossNumericDerivativeSolutions[index]).toFixed(4) : "");
                            // update state
                            this.setState({ weights: newWeights, weightsHistory: weightsHistory, lossHistory: lossHistory, predictionHistory: predictionHistory });
                        }}>Train</button>
                        <button type="button" className="button" onClick={() => {
                            // reset weights
                            if (this.state.weightsHistory.length > 0) this.setState({ weights: this.state.weightsHistory[0], weightsHistory: [], lossHistory: [], predictionHistory: [] });
                        }}>Reset</button>
                        {partialLossNumericDerivativeSolutions.map((item, index) => {
                            var newWeight = this.state.weights[index] - this.state.learningRate * partialLossNumericDerivativeSolutions[index];
                            if (item) {
                                return (
                                    <div>
                                        <MathJax.Context input='ascii'>
                                            <div>
                                                <MathJax.Node>{"w_" + index + "=" + "w_" + index + "-" + this.state.learningRate + "*" + "loss'_{w_" + index + "}=" + parseFloat(newWeight.toFixed(4))}</MathJax.Node>
                                            </div>
                                        </MathJax.Context>
                                    </div>
                                )
                            }
                        })}
                    </div>
                    <div className="grid-item">
                        <div className="grid-title">Visualization
                        <div style={{ display: "inline", float: "right" }} onChange={event => this.setState({ radio: event.target.value })}>
                                <input style={{ margin: "0 5px" }} type="radio" id="loss" name="visualization" value="loss" defaultChecked />
                                <label style={{ margin: "0 5px" }} for="loss">Loss </label>
                                <input style={{ margin: "0 5px" }} type="radio" id="weights" name="visualization" value="weights" />
                                <label style={{ margin: "0 5px" }} for="weights">Weights </label>
                                <input style={{ margin: "0 5px" }} type="radio" id="prediction" name="visualization" value="prediction" />
                                <label style={{ margin: "0 5px" }} for="prediction">Prediction </label>
                            </div>
                        </div>
                        {(() => {
                            switch (this.state.radio) {
                                case 'loss':
                                    return (
                                        <VictoryChart domain={{ x: [0, Math.max(10, this.state.lossHistory.length)], y: [Math.min(Math.min(Math.min(...this.state.lossHistory), parseFloat(lossNumericSolution)), 0), Math.max(Math.max(...this.state.lossHistory), parseFloat(lossNumericSolution))] }}>
                                            <VictoryAxis label="iterations" />
                                            <VictoryAxis dependentAxis />
                                            <VictoryScatter data={[{ x: this.state.lossHistory.length, y: lossNumericSolution }]} size={5} style={{ data: { fill: "#c43a31" } }} />
                                            {this.state.lossHistory.map((item, index) => {
                                                return <VictoryScatter data={[{ x: index, y: item }]} size={5} style={{ data: { fill: "#c43a31" } }} />
                                            })}
                                            <VictoryLine style={{ data: { stroke: "#c43a31" } }} data={[{ x: this.state.lossHistory.length - 1, y: this.state.lossHistory[this.state.lossHistory.length - 1] }, { x: this.state.lossHistory.length, y: lossNumericSolution },]} />
                                            {this.state.lossHistory.map((item, index, array) => {
                                                if (index > 0) {
                                                    return <VictoryLine style={{ data: { stroke: "#c43a31" } }} data={[
                                                        { x: index - 1, y: array[index - 1] },
                                                        { x: index, y: array[index] },
                                                    ]} />
                                                }
                                            })}
                                        </VictoryChart>
                                    );
                                case 'weights':
                                    return (
                                        <VictoryChart domain={{ x: [0, Math.max(10, this.state.weightsHistory.length)], y: [Math.min(Math.min(Math.min(...(this.state.weightsHistory.flat(Infinity).map((item) => isNaN(parseFloat(item)) ? 0 : parseFloat(item)))), ...this.state.weights), 0), Math.max(Math.max(...(this.state.weightsHistory.flat(Infinity).map((item) => isNaN(parseFloat(item)) ? 0 : parseFloat(item)))), ...this.state.weights)] }}>
                                            <VictoryAxis label="iterations" />
                                            <VictoryAxis dependentAxis />
                                            {this.state.weights.map((item, index) => {
                                                return <VictoryScatter data={[{ x: this.state.weightsHistory.length, y: parseFloat(item) }]} size={5} style={{ data: { fill: "green" } }} />
                                            })}
                                            {this.state.weightsHistory.map((outerItem, outerIndex) => {
                                                return outerItem.map((item, index) => {
                                                    return <VictoryScatter data={[{ x: outerIndex, y: parseFloat(item) }]} size={5} style={{ data: { fill: "green" } }} />
                                                })
                                            })}
                                            {this.state.weights.map((item, index) => {
                                                if (this.state.weightsHistory.length > 0) {
                                                    return <VictoryLine style={{ data: { stroke: "green" } }} data={[
                                                        { x: this.state.weightsHistory.length - 1, y: parseFloat(this.state.weightsHistory[this.state.weightsHistory.length - 1][index]) },
                                                        { x: this.state.weightsHistory.length, y: parseFloat(item) },
                                                    ]} />
                                                }
                                            })}
                                            {this.state.weightsHistory.map((outerItem, outerIndex, array) => {
                                                return outerItem.map((item, index) => {
                                                    if (outerIndex > 0) {
                                                        return <VictoryLine style={{ data: { stroke: "green" } }} data={[
                                                            { x: outerIndex - 1, y: parseFloat(array[outerIndex - 1][index]) },
                                                            { x: outerIndex, y: parseFloat(array[outerIndex][index]) },
                                                        ]} />
                                                    }
                                                })
                                            })}
                                        </VictoryChart>
                                    );
                                case 'prediction':
                                    return (
                                        <VictoryChart domain={{ x: [0, Math.max(10, this.state.predictionHistory.length)], y: [Math.min(Math.min(Math.min(Math.min(...this.state.predictionHistory), parseFloat(formulaNumericSolution)), parseFloat(this.state.t)), 0), Math.max(Math.max(Math.max(Math.max(...this.state.predictionHistory), parseFloat(formulaNumericSolution)), parseFloat(this.state.t)), 0)] }}>
                                            <VictoryAxis label="iterations" />
                                            <VictoryAxis dependentAxis />
                                            <VictoryScatter data={[{ x: this.state.predictionHistory.length, y: formulaNumericSolution }]} size={5} style={{ data: { fill: "blue" } }} />
                                            {this.state.predictionHistory.map((item, index) => {
                                                return <VictoryScatter data={[{ x: index, y: item }]} size={5} style={{ data: { fill: "blue" } }} />
                                            })}
                                            <VictoryLine style={{ data: { stroke: "blue" } }} data={[{ x: this.state.predictionHistory.length - 1, y: this.state.predictionHistory[this.state.predictionHistory.length - 1] }, { x: this.state.predictionHistory.length, y: formulaNumericSolution },]} />
                                            {this.state.predictionHistory.map((item, index, array) => {
                                                if (index > 0) {
                                                    return <VictoryLine style={{ data: { stroke: "blue" } }} data={[
                                                        { x: index - 1, y: array[index - 1] },
                                                        { x: index, y: array[index] },
                                                    ]} />
                                                }
                                            })}
                                            <VictoryLine style={{ data: { stroke: "blue" } }} data={[
                                                { x: 0, y: parseFloat(this.state.t) },
                                                { x: this.state.predictionHistory.length, y: parseFloat(this.state.t) },
                                            ]} />
                                        </VictoryChart>
                                    );
                                default:
                                    return "";
                            }
                        })()}
                    </div>
                </div>
                <div className="grid-container-2">
                    <div className="grid-item" style={{ overflowX: 'scroll', overflowY: 'hide', paddingBottom: '5px' }}>
                        <div className="grid-title">Derivatives</div>
                        {partialLossSymbolicEquations.map((item, index) => {
                            if (item != "") {
                                return (
                                    <div>
                                        <MathJax.Context input='ascii'>
                                            <div>
                                                <p></p>
                                                <MathJax.Node>{"loss_{w_" + index + "}=" + item}</MathJax.Node>
                                                <p></p>
                                                <MathJax.Node>{"loss'_{w_" + index + "}=" + partialLossSymbolicDerivatives[index] + "=" + parseFloat(partialLossNumericDerivativeSolutions[index].toFixed(4))}</MathJax.Node>
                                            </div>
                                        </MathJax.Context>
                                    </div>
                                )
                            }
                        })}
                    </div>
                </div>
            </div>
        );
    }
}

module.exports = ExerciseCreator;