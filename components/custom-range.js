import React from 'react';

class CustomRange extends React.PureComponent {
  constructor(props) {
    super(props);
  }

  handleChange(event) {
    this.props.updateProps({
      value: +event.target.value
    });
  }

  render() {
    const { value, min, max, step, className, style } = this.props;
    return (
      <input
        type="range"
        onChange={this.handleChange.bind(this)}
        className={className}
        value={value}
        min={min}
        max={max}
        step={step}
        style={style}
      />
    );
  }
}

CustomRange.defaultProps = {
  value: 0,
  min: 0,
  max: 1,
  step: 1
};

CustomRange._idyll = {
  name: 'Range',
  tagType: 'closed',
  props: [
    {
      name: 'value',
      type: 'number',
      example: 'x',
      description:
        'The value to display; if this is a variable, the variable will automatically be updated when the slider is moved.'
    },
    {
      name: 'min',
      type: 'number',
      example: '0',
      description: 'The minimum value.'
    },
    {
      name: 'max',
      type: 'number',
      example: '100',
      description: 'The maximum value.'
    },
    {
      name: 'step',
      type: 'number',
      example: '1',
      defaultValue: '1',
      description: 'The granularity of the slider.'
    }
  ]
};

export default CustomRange;