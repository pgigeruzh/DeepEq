import * as tf from '@tensorflow/tfjs';
const React = require('react');

class PixelVisualization extends React.PureComponent {
    static defaultProps = {
        numberOfLayers: 1,
        maxNumberOfLayers: 100,
        hideTrainButton: false,
        hideAddLayer: false,
        hideRemoveLayer: false,
        hideLearningRate: false,
    }

    constructor(props) {
        super(props);
        // bind functions
        this.customModelLoss = this.customModelLoss.bind(this);
        this.initModel = this.initModel.bind(this);
        this.trainModel = this.trainModel.bind(this);
        this.predictModel = this.predictModel.bind(this);
        this.getModelWeights = this.getModelWeights.bind(this);
        this.setModelWeights = this.setModelWeights.bind(this);
        // run functions
        var numberOfLayers = 0;
        this.props.numberOfLayers ? numberOfLayers = this.props.numberOfLayers : numberOfLayers = 1;
        this.model = this.initModel(numberOfLayers, 0.001);
        this.setModelWeights(this.model, Array(numberOfLayers).fill(1));
        this.trainModel(this.model, 150, 200);
        // set states
        var prediction = this.predictModel(this.model, 150);
        this.state = {
            t: 200,
            x: 150,
            weights: this.getModelWeights(this.model),
            y: prediction,
            loss: this.customModelLoss(tf.tensor(prediction[prediction.length - 1]), tf.tensor(200)).arraySync(),
            learningRate: 0.001
        }
    }

    customModelLoss(yPred, yTrue) {
        return tf.square(yTrue.sub(yPred));
    }

    initModel(numberOfLayers, learningRate) {
        // create model
        var layers = [tf.layers.dense({ inputShape: 1, units: 1, useBias: false, activation: 'linear' })];
        for (var i = 1; i < numberOfLayers; i++) {
            layers.push(tf.layers.dense({ units: 1, useBias: false, activation: 'linear' }));
        }
        var model = tf.sequential({
            layers: layers
        });
        // compile model
        model.compile({
            optimizer: tf.train.sgd(learningRate*0.001),
            loss: this.customModelLoss,
            metrics: ['accuracy']
        });
        return model;
    }

    setModelWeights(model, weights) {
        for (var i = 0; i < model.layers.length; i++) {
            model.layers[i].setWeights([tf.tensor(parseFloat(weights[i]), model.layers[i].getWeights()[0].shape)]);
        }
    }

    getModelWeights(model) {
        var weights = [];
        for (var i = 0; i < model.layers.length; i++) {
            weights.push(model.layers[i].getWeights()[0].arraySync());
        }
        return weights;
    }

    trainModel(model, x, t) {
        // create data
        const data = tf.tensor([[x]]);
        const labels = tf.tensor([[t]]);

        // train model
        model.fit(data, labels, {
            epochs: 10,
        });
    }

    predictModel(model, input) {
        // get intermediary layer output
        var input = tf.tensor([[parseInt(input)]]);
        var layers = model.layers;
        var result = [];
        for (var i = 0; i < layers.length; i++) {
            var layer = layers[i];
            var output = layer.apply(input);
            input = output;
            // flatten output
            var output_flat = [].concat(...output.arraySync());
            result.push(output_flat[0]);
        }
        return result;
    }

    changeWeight(index, value) {
        // update state
        var weights = [...this.state.weights];
        weights[index] = parseFloat(value);
        this.setState({
            weights: weights
        });
        // update tensorflow weights
        this.setModelWeights(this.model, weights);
        // update prediction
        var prediction = this.predictModel(this.model, this.state.x);
        this.setState({ y: prediction, loss: this.customModelLoss(tf.tensor(Math.round(prediction[prediction.length - 1])), tf.tensor(this.state.t)).arraySync() });
    }

    render() {
        return (
            <div>
                <svg viewBox={(-300 + this.state.weights.length * 100) + " 0 1100 250"}>
                    <rect fill={"rgb(" + this.state.x + "," + this.state.x + "," + this.state.x + ")"} height="100" width="100" y="50" x="50" strokeWidth="1.5" stroke="#000" />
                    <text textAnchor="start" fontFamily="Helvetica, Arial, sans-serif" fontSize="40" y="36" x="90" strokeWidth="0" stroke="#000" fill="#000000">x</text>
                    <text fill={"rgb(" + (this.state.x < 120 ? 255 : 0) + "," + (this.state.x < 120 ? 255 : 0) + "," + (this.state.x < 120 ? 255 : 0) + ")"} textAnchor="start" fontFamily="Helvetica, Arial, sans-serif" fontSize="40" y="115" x="70">{Number(this.state.x).toFixed(0)}</text>
                    <line y2="1331" x2="1402.5" y1="1326" x1="1241.5" strokeWidth="1.5" stroke="#000" fill="none" />
                    {this.state.weights.map((item, i) => (
                        <svg>
                            <line y1="100" x1={150 + i * 200} y2="100" x2={250 + i * 200} strokeWidth="1.5" stroke="#000" />
                            <rect fill={"rgb(" + this.state.y[i] + "," + this.state.y[i] + "," + this.state.y[i] + ")"} height="100" width="100" y="50" x={250 + i * 200} strokeWidth="1.5" stroke="#000" />
                            <text textAnchor="start" fontFamily="Helvetica, Arial, sans-serif" fontSize="40" y="70" x={180 + i * 200} strokeWidth="0" stroke="#000" fill="#000000">
                                w<tspan dy="10" fontSize="25">{i}</tspan>
                            </text>
                            <text textAnchor="start" fontFamily="Helvetica, Arial, sans-serif" fontSize="40" y="140" x={160 + i * 200} strokeWidth="0" stroke="#000">{Number(item).toFixed(2)}</text>
                            <text textAnchor="start" fill={"rgb(" + (this.state.y[i] < 120 ? 255 : 0) + "," + (this.state.y[i] < 120 ? 255 : 0) + "," + (this.state.y[i] < 120 ? 255 : 0) + ")"} fontFamily="Helvetica, Arial, sans-serif" fontSize="40" y="115" x={270 + i * 200}>{Number(this.state.y[i]).toFixed(0)}</text>
                            <text textAnchor="start" fontFamily="Helvetica, Arial, sans-serif" fontSize="40" y="210" x={140 + (this.state.weights.length - 1) * 50} strokeWidth="0" stroke="#000" fill="#000000">
                                <tspan wordSpacing={this.state.weights.length * 10}>y = </tspan>
                                <tspan dx={-i * 10}>(</tspan>
                                <tspan dx={i * 10}>x</tspan>
                                <tspan dx={i * 60}>w</tspan>
                                <tspan dy="10" fontSize="25">{i}</tspan>
                                <tspan dy="-10">)</tspan>
                            </text>
                        </svg>
                    ))}
                    <text textAnchor="start" fontFamily="Helvetica, Arial, sans-serif" fontSize="40" y="210" x={500 + (this.state.weights.length - 1) * 100} strokeWidth="0" stroke="#000" fill="#000000">
                        <tspan>loss = </tspan>
                        <tspan>{Number(this.state.loss).toFixed(0)}</tspan>
                    </text>
                    <text textAnchor="start" fontFamily="Helvetica, Arial, sans-serif" fontSize="40" y="35" x={295 + (this.state.weights.length - 1) * 200} strokeWidth="0" stroke="#000" fill="#000000">y</text>
                    <text textAnchor="start" fontFamily="Helvetica, Arial, sans-serif" fontSize="40" y="35" x={495 + (this.state.weights.length - 1) * 200} strokeWidth="0" stroke="#000" fill="#000000">t</text>
                    <rect fill={"rgb(" + this.state.t + "," + this.state.t + "," + this.state.t + ")"} height="100" width="100" y="50" x={450 + (this.state.weights.length - 1) * 200} strokeWidth="1.5" stroke="#000" />
                    <text fill={"rgb(" + (this.state.t < 120 ? 255 : 0) + "," + (this.state.t < 120 ? 255 : 0) + "," + (this.state.t < 120 ? 255 : 0) + ")"} textAnchor="start" fontFamily="Helvetica, Arial, sans-serif" fontSize="40" y="115" x={470 + (this.state.weights.length - 1) * 200}>{Number(this.state.t).toFixed(0)}</text>
                </svg>
                <div>
                    <button type="button" className="button" style={{ display: this.props.hideTrainButton ? "none" : "inline" }} onClick={() => {
                        this.trainModel(this.model, this.state.x, this.state.t);
                        var prediction = this.predictModel(this.model, this.state.x);
                        this.setState({
                            y: prediction,
                            weights: this.getModelWeights(this.model),
                            loss: this.customModelLoss(tf.tensor(Math.round(prediction[prediction.length - 1])), tf.tensor(this.state.t)).arraySync(),
                        });
                    }}>Train (10 Steps)</button>
                    <button type="button" className="button" style={{ display: this.props.hideAddLayer ? "none" : "inline" }} onClick={(e) => {
                        if (this.state.weights.length < this.props.maxNumberOfLayers - 1) {
                            this.model = this.initModel(this.state.weights.length + 1, this.state.learningRate);
                            this.setModelWeights(this.model, [...this.state.weights, 1.0]);
                            var prediction = this.predictModel(this.model, this.state.x);
                            this.setState({
                                y: prediction,
                                weights: this.getModelWeights(this.model),
                                loss: this.customModelLoss(tf.tensor(Math.round(prediction[prediction.length - 1])), tf.tensor(this.state.t)).arraySync(),
                            });
                        }
                    }}>Add Layer</button>
                    <button type="button" className="button" style={{ display: this.props.hideRemoveLayer ? "none" : "inline" }} onClick={(e) => {
                        this.model = this.initModel(this.state.weights.length - 1, this.state.learningRate);
                        this.setModelWeights(this.model, this.state.weights);
                        var prediction = this.predictModel(this.model, this.state.x);
                        this.setState({
                            y: this.predictModel(this.model, this.state.x),
                            weights: this.getModelWeights(this.model),
                            loss: this.customModelLoss(tf.tensor(Math.round(prediction[prediction.length - 1])), tf.tensor(this.state.t)).arraySync(),
                        });
                    }}>Remove Layer</button>
                    <div style={{ float: "right" }}>
                        <label htmlFor="learningRate" style={{ display: this.props.hideLearningRate ? "none" : "inline" }}>learning rate: </label>
                        <input id="learningRate" type="text" style={{ width: "100px", display: this.props.hideLearningRate ? "none" : "inline" }} value={this.state.learningRate} onChange={(e) => {
                            this.model = this.initModel(this.state.weights.length, parseFloat(e.target.value));
                            this.setState({ learningRate: parseFloat(e.target.value) });
                        }} />
                    </div>
                </div>
                <label htmlFor="sliderX">x: </label>
                <input id="sliderX" className="slider" type="range" min="0" max="255" value={this.state.x} onChange={(e) => {
                    var prediction = this.predictModel(this.model, e.target.value);
                    this.setState({ x: parseFloat(e.target.value), y: prediction, loss: this.customModelLoss(tf.tensor(Math.round(prediction[prediction.length - 1])), tf.tensor(this.state.t)).arraySync() })
                }} step="1" />
                {this.state.weights.map((item, i) => (
                    <div style={{ display: "inline" }}>
                        <label htmlFor={"sliderW" + i}>w{i}: </label>
                        <input id={"sliderW" + i} className="slider" type="range" min="0" max="2" value={this.state.weights[i]} onChange={(e) => this.changeWeight(i, e.target.value)} step="0.01" />
                    </div>
                ))}
                <div>

                    <label htmlFor="sliderT">t: </label>
                    <input id="sliderT" className="slider" type="range" min="0" max="255" value={this.state.t} onChange={(e) => {
                        var prediction = this.predictModel(this.model, this.state.x);
                        this.setState({ t: parseInt(e.target.value), loss: this.customModelLoss(tf.tensor(Math.round(prediction[prediction.length - 1])), tf.tensor(parseInt(e.target.value))).arraySync(), })
                    }} step="1" />
                </div>
            </div>
        );
    }
}

module.exports = PixelVisualization;