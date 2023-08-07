import "./ListGroup.css";

function ListGroup({ items }) {
  return (
    <ul className="item-group w-50 pb-10 pt-6">
      {items.map((item) => (
        <li key={item.id} className="list-group-item border-0 pb-3">
          {item.item}
        </li>
      ))}
    </ul>
  );
}

export default ListGroup;
