import fetch from 'isomorphic-fetch'
import {useState} from 'react'
import './signup.css'
const SignUp = () => {
    const [ email, setEmail] = useState("")
    function Inputs(event){
        console.log(event.target.value)
        setEmail({ 
            
            name:event.target.value
        });
    }
   function Addemail(){
       console.log(email)
       return(
        fetch('http://localhost:3001/emails', {
            method:'POST',
            mode: 'cors',
            body: JSON.stringify({
                "name":"email"
            }),
            headers: {
                'Content-Type': 'application/json'
            }
        }).then(response =>{
            if (response.status==200){
                return response;
            }
            else{
                console.log('BadEmail');
            }
        })
       )
    }
    return (
        <div className="signup">          
            <h1 >Sign up for our mailing list!</h1>
        <form onSubmit={Addemail} className="form" >
         <input type="email" name="email" value={email.name} onChange={Inputs} placeholder="Email here" className="email"></input>
         <input type="submit" name="submit" className="btn"/>
        </form>
        </div>
    )

}
export default SignUp
