const React = require('react');

class SvgTSpan extends React.PureComponent {

    constructor(props) {
        super(props);
    }

    render() {
        return (
            <tspan {...this.props}>
                {this.props.value}{this.props.children}
            </tspan>
        );
    }
}

module.exports = SvgTSpan;