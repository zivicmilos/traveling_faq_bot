import './App.css';
import ListGroup from '../ListGroup/ListGroup'
import QuestionBox from '../QuestionBox/QuestionBox'
import Button from '../Button/Button'
import {getAnswer} from '../../services/faq-service'
import axios from "axios";

 const handleSubmit = (e) => {
    e.preventDefault();
    const question = {
      question: 'What Are Rate For Long Term Care Insurance?',
    };
    axios.post('http://localhost:8000/faq/questions', question).then((response) => {
      console.log(response.status, response.data);
    });
  };

function App() {
  return <div>
  	<ListGroup></ListGroup>
  	<QuestionBox></QuestionBox>
  	<Button onClick={getAnswer('asd').apply()}>Submit</Button>
  </div>;
}

export default App;
