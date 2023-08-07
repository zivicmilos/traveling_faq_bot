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
  const [question, setQuestion] = useState("");
  const [items, setItems] = useState(initialState);
  const [model, setModel] = useState("custom");
  const [preprocessing, setPreprocessing] = useState(false);
  const [weight, setWeight] = useState(true);

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
      preprocessing: preprocessing,
      weight: weight,
    };
    setItems((current) => [
      ...current,
      { id: items.length, item: "Q: " + question },
    ]);
    getAnswer(q).then((answer) => {
      console.log(answer);
      setItems((current) => [
        ...current,
        { id: items.length + 1, item: "A: " + answer },
      ]);
    });

    setQuestion("");
  };

  const handleReset = () => {
    setItems([]);
  };

  const onChangeModel = (e) => {
    console.log(e.target.value);
    if (e.target.value === "tf" || e.target.value === "tf-idf") {
      setPreprocessing(true);
      setWeight(false);
    } else if (e.target.value === "custom" || e.target.value === "pretrained") {
      setPreprocessing(false);
      setWeight(true);
    } else {
      setPreprocessing(false);
      setWeight(false);
    }
    setModel(e.target.value);
  };

  const onChangePreprocessing = (e) => {
    console.log(e.target.value);
    setPreprocessing(e.target.value);
  };

  const onChangeWeight = (e) => {
    console.log(e.target.value);
    setWeight(e.target.value);
  };

  return (
    <div className="App">
      <Navbar handleReset={handleReset} />
      <div className="hstack">
        <RadioButtons
          preprocessing={preprocessing}
          weight={weight}
          onChangeModel={onChangeModel}
          onChangePreprocessing={onChangePreprocessing}
          onChangeWeight={onChangeWeight}
        />
        {!!items.length && <ListGroup items={items} />}
        {!items.length && (
          <div className="d-flex justify-content-center pt-25 w-50">
            <div>Feel free to ask any insurance-related question :)</div>
          </div>
        )}
      </div>
      <div className="fixed-bottom justify-content-center hstack gap-3 pb-3 pt-3 bg-body-tertiary">
        <QuestionBox
          question={question}
          setQuestion={setQuestion}
          onEnter={handleSubmit(question)}
        />
      </div>
    </div>
  );
}

export default App;
