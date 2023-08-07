import "./App.css";
import Navbar from "../Navbar/Navbar";
import ListGroup from "../ListGroup/ListGroup";
import QuestionBox from "../QuestionBox/QuestionBox";
import RadioButtons from "../RadioButtons/RadioButtons";
import { useStore } from "../../helpers/store";

function App() {
  const items = useStore((state) => state.items);

  return (
    <div className="App">
      <Navbar />
      <div className="hstack">
        <RadioButtons />
        {!!items.length && <ListGroup />}
        {!items.length && (
          <div className="d-flex justify-content-center pt-25 w-50">
            <div>Feel free to ask any insurance-related question :)</div>
          </div>
        )}
      </div>
      <div className="fixed-bottom justify-content-center hstack gap-3 pb-3 pt-3 bg-body-tertiary">
        <QuestionBox />
      </div>
    </div>
  );
}

export default App;
