function ListGroup() {
	let items = ['Question 1', 'Answer 1', 'Question 2', 'Answer 2'];

	return <ul className="list-group">
		{items.map(item => <li key={item} className="list-group-item border-0">{item}</li>)}
	</ul>
}

export default ListGroup;