import React, { Component } from 'react';
import PropTypes from 'prop-types';
import '../style/EmojiDisplay.css';

const mapping = {
	'0': '❤',
	'1': '😍',
	'2': '😂',
	'3': '💕',
	'4': '🔥',
	'5': '😊',
	'6': '😎',
	'7': '✨',
	'8': '💙',
	'9': '😘',
	'10': '📷',
	'11': '🇺🇸',
	'12': '☀',
	'13': '💜',
	'14': '😉',
	'15': '💯',
	'16': '😁',
	'17': '🎄',
	'18': '📸',
	'19': '😜'
};

export default class EmojiDisplay extends Component {
	render() {
		return (
			<div className="emoji-container">
				{this.props.labels
					.slice(0, (this.props.numberOfEmojis || 3) + 1)
					.map(({ label }) => <span key={label}>{mapping[label]}</span>)}
			</div>
		);
	}
}

EmojiDisplay.propTypes = {
	numberOfEmojis: PropTypes.number,
	labels: PropTypes.array.isRequired
};
