import "./QuestionBox.css";

function QuestionBox({ question, setQuestion, onEnter }) {
  const handleChange = (e) => setQuestion(e.target.value);

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
