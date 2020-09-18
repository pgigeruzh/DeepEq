import React from 'react';
import { VictoryChart, VictoryAxis, VictoryLine, VictoryScatter } from 'victory';

class GradientVisualization extends React.PureComponent {
  render() {
    var weightLabel = "Weight (w = " + Number(this.props.weights[this.props.weights.length - 1]).toFixed(2) + ")";
    return (
      <VictoryChart domain={{ x: [-1, 1], y: [0, 1] }}>
        <VictoryAxis label={weightLabel} />
        <VictoryAxis label="Loss" dependentAxis offsetY={100} />
        <VictoryLine samples={200} y={(d) => Math.pow(d.x, 2)} />
        {this.props.weights.map((item, i, array) => {
          if (i > 0) {
            return <VictoryLine style={{ data: { stroke: "#c43a31" } }} data={[
              { x: array[i - 1], y: Math.pow(array[i - 1], 2) },
              { x: array[i], y: Math.pow(array[i], 2) },
            ]} />
          }
        })}
        <VictoryScatter data={[{ x: this.props.weights[this.props.weights.length - 1], y: Math.pow(this.props.weights[this.props.weights.length - 1], 2) }]} size={15} style={{ data: { fill: "#c43a31" } }} />
      </VictoryChart>
    );
  }
}

export default GradientVisualization;