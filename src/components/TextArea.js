import React, { Component, Fragment } from 'react';
import '../style/TextArea.css';
import PropTypes from 'prop-types';

export default class TextArea extends Component {
	constructor(props) {
		super(props);
		this.state = {
			tweet: this.props.defaultValue || '#tweet'
		};
		this.mirror = React.createRef();
	}

	componentDidMount() {
		this.updateBoundings(this.state.tweet);
	}

	updateBoundings(tweet) {
		let mirror = this.mirror.current;
		mirror.innerHTML = tweet;
		const maxWidth = 0.75 * window.innerWidth;
        const width = mirror.clientWidth + 20;
		const heightMultiply = parseInt(width / maxWidth) + 1;
		this.setState({ width: width < maxWidth ? width : maxWidth, height: 1.3 * heightMultiply + 'em' });
	}

	onChange(event) {
		const tweet = event.target.value.replace('\n', '');
		this.setState({ tweet });
		this.updateBoundings(tweet);
		this.props.onChange(tweet);
	}

	handleBlur() {
		if (this.state.tweet.length === 0) {
			const tweet = this.props.defaultValue || '#tweet';
			this.setState({ tweet });
			this.updateBoundings(tweet);
		}
	}

	render() {
		return (
			<Fragment>
				<div className="textarea-container">
					<textarea
						style={{ width: this.state.width, height: this.state.height }}
						type="text"
						id="textarea"
						onChange={this.onChange.bind(this)}
						value={this.state.tweet}
						onBlur={this.handleBlur.bind(this)}
					/>
				</div>
				<div className="textarea-mirror" ref={this.mirror}>
					{this.state.tweet}
				</div>
			</Fragment>
		);
	}
}

TextArea.propTypes = {
	defaultValue: PropTypes.string,
	onChange: PropTypes.func.isRequired
};
