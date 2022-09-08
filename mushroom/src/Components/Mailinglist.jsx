import react, {useState} from 'react'
import SignUp from '../Components/SignUp'
import ThankYou from '../Components/ThankYou'
const Mailinglist=( )=>{
    const [isSubmitted, setIsSubmitted] = useState(false);
    function onSubmit(){
        setIsSubmitted(true);
    }
return(
<div>
<h1>
Your Email please    
</h1>
{!isSubmitted ?
    <SignUp onSubmit={onSubmit}/>:
    <ThankYou/>
}

</div>



)


}
export default Mailinglist;