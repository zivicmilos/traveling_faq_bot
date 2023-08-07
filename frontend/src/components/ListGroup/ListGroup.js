import "./ListGroup.css";
import { useStore } from "../../helpers/store";

function ListGroup() {
  const items = useStore((state) => state.items);

  return (
    <ul className="item-group w-50 pb-10 pt-20">
      {items.map((item) => (
        <li key={item.id} className="list-group-item border-0 pb-3">
          {item.item}
        </li>
      ))}
    </ul>
  );
}

export default ListGroup;
