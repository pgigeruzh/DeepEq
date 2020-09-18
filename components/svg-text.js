const React = require('react');

class SvgText extends React.PureComponent {

    constructor(props) {
        super(props);
    }

    render() {
        return (
            <svg>
                <text {...this.props}>{this.props.value}{this.props.children}</text>
            </svg>
        );
    }
}

module.exports = SvgText;