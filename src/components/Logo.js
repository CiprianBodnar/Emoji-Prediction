import React, { Component } from 'react';
import '../style/Logo.css';
import FontAwesome from 'react-fontawesome';
export default class Logo extends Component {
	render() {
		return (
			<div className="logo-container">
				<span id="twitter">
					<FontAwesome name="twitter" />
				</span>
				<span id="two">
					<FontAwesome name="long-arrow-right" />
				</span>
				<span id="emoji">
					<FontAwesome name="smile-o" />
				</span>
			</div>
		);
	}
}
