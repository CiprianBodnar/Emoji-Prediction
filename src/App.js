import React, { Component, Fragment } from 'react';
import './style/App.css';

import Logo from './components/Logo';
import TextArea from './components/TextArea';
import EmojiDisplay from './components/EmojiDisplay';

const API_URL = '/api/emoji?tweet=';

export default class App extends Component {
	constructor(props) {
		super(props);
		this.state = { emojis: [] };
	}

	parseResponse(xmlDoc) {
		let emojis = [];
		for (let emoji of xmlDoc.getElementsByTagName('emoji')) {
			const label = emoji.getAttribute('label');
			const position = emoji.getAttribute('position');
			const name = emoji.getAttribute('emoji_name');
			emojis[position] = { name, label };
		}
		this.setState({ emojis });
	}

	onInput(input) {
		console.log(input);
		let parser = new DOMParser();
		fetch(API_URL + encodeURIComponent(input))
			.then((response) => response.text())
			.then((text) => parser.parseFromString(text, 'text/xml'))
			.then((xml) => this.parseResponse(xml));
	}

	render() {
		return (
			<Fragment>
				<header id="header-bar">
					<Logo />
				</header>
				<section id="content">
					<TextArea defaultValue="enter your #tweet here" onChange={this.onInput.bind(this)} />
				</section>
				<section id="emojis">
					<EmojiDisplay labels={this.state.emojis} />
				</section>
				<footer>
					<span>#Tweet2Emoji</span> project by <span>Ciprian Bodnar</span> & <span>Cristian Hasna</span>
				</footer>
			</Fragment>
		);
	}
}
