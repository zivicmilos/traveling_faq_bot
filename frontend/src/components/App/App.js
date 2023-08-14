import "./App.css";
import Navbar from "../Navbar/Navbar";
import ListGroup from "../ListGroup/ListGroup";
import QuestionBox from "../QuestionBox/QuestionBox";
import RadioButtons from "../RadioButtons/RadioButtons";
import { useStore } from "../../helpers/store";

function App() {
  const items = useStore((state) => state.items);
  const defaultModel = useStore((state) => state.defaultModel);

  return (
    <div className="App">
      <Navbar />
      <div className="CentralBox hstack">
        {defaultModel === "customized" ? (
          <RadioButtons />
        ) : (
          <div className="w-25 px-3"></div>
        )}
        {!!items.length && <ListGroup />}
        {!items.length && (
          <div className="w-50 hstack justify-content-center">
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
