import "./QuestionBox.css";
import { useEffect, useState } from "react";

function QuestionBox({ resetTextArea, setData, onEnter }) {
  const [question, setQuestion] = useState("");
  const handleChange = (e) => setQuestion(e.target.value);

  useEffect(() => {
    setQuestion("");
  }, [resetTextArea]);

  useEffect(() => {
    setData(question);
  });

  return (
    <textarea
      className="form-control w-50 p-3"
      placeholder="Here you can input you question..."
      onChange={handleChange}
      value={question}
      onKeyDown={onEnter}
    ></textarea>
  );
}

export default QuestionBox;
