import "./QuestionBox.css";
import { useStore } from "../../helpers/store";
import {
  getAnswerCustomized,
  getAnswerDefault,
} from "../../services/faq-service";
import { storeData } from "../../helpers/local-storage";
import { useEffect } from "react";

function QuestionBox() {
  const items = useStore((state) => state.items);
  const addItem = useStore((state) => state.addItem);
  const question = useStore((state) => state.question);
  const setQuestion = useStore((state) => state.setQuestion);
  const model = useStore((state) => state.model);
  const preprocessing = useStore((state) => state.preprocessing);
  const weight = useStore((state) => state.weight);
  const defaultModel = useStore((state) => state.defaultModel);

  const handleChange = (e) => setQuestion(e.target.value);

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
    let item = { id: items.length, item: "Q: " + question };
    addItem(item);

    if (defaultModel === "default") {
      getAnswerDefault(q).then((answer) => {
        console.log(answer);
        let item = { id: items.length + 1, item: "A: " + answer };
        addItem(item);
      });
    } else if (defaultModel === "customized") {
      getAnswerCustomized(q).then((answer) => {
        console.log(answer);
        let item = { id: items.length + 1, item: "A: " + answer };
        addItem(item);
      });
    }

    setQuestion("");
  };

  useEffect(() => {
    storeData("items", items);
  }, [items]);

  return (
    <textarea
      className="form-control w-50 p-3"
      placeholder="Here you can input you question..."
      onChange={handleChange}
      value={question}
      onKeyDown={handleSubmit(question)}
    ></textarea>
  );
}

export default QuestionBox;
