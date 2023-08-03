import "./App.css";
import Navbar from "../Navbar/Navbar";
import ListGroup from "../ListGroup/ListGroup";
import QuestionBox from "../QuestionBox/QuestionBox";
import { getAnswer } from "../../services/faq-service";
import { getData, storeData } from "../../helpers/local-storage";
import { useState, useEffect } from "react";
import RadioButtons from "../RadioButtons/RadioButtons";

function App() {
  const initialState = () => getData("items") || [];
  const [data, setData] = useState("");
  const [items, setItems] = useState(initialState);
  const [resetTextArea, setResetTextArea] = useState(false);
  const [model, setModel] = useState("pretrained");

  useEffect(() => {
    storeData("items", items);
  }, [items]);

  const handleSubmit = (question) => (e) => {
    if (e.key !== "Enter") return;

    e.preventDefault();
    console.log(question);
    const q = {
      question: question,
      model: model,
    };
    setItems((current) => [...current, "Q: " + question]);
    getAnswer(q).then((data) => {
      console.log(data);
      setItems((current) => [...current, "A: " + data]);
    });

    setResetTextArea(!resetTextArea);
  };

  const handleReset = (e) => {
    setItems([]);
  };

  const onChangeValue = (e) => {
    console.log(e.target.value);
    setModel(e.target.value);
  };

  return (
    <div className="App">
      <Navbar handleReset={handleReset} />
      <div className="hstack">
        <RadioButtons onChangeValue={onChangeValue} />
        {items.length !== 0 && <ListGroup items={items} />}
        {items.length === 0 && (
          <div className="d-flex justify-content-center pt-25 w-50">
            <div>Feel free to ask any insurance-related question :)</div>
          </div>
        )}
      </div>
      <div className="fixed-bottom justify-content-center hstack gap-3 pb-3 pt-3 bg-body-tertiary">
        <QuestionBox
          resetTextArea={resetTextArea}
          setData={setData}
          onEnter={handleSubmit(data)}
        />
      </div>
    </div>
  );
}

export default App;
