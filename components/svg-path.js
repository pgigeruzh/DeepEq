const React = require('react');

class SvgPath extends React.PureComponent {

    constructor(props) {
        super(props);
    }

    render() {
        return (
            <svg>
                <path {...this.props} />
            </svg>
        );
    }
}

module.exports = SvgPath;