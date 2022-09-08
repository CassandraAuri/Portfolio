import {exelist} from './Exe-list'
import enoki from '../Components/Photos/enoki.jpg'
import kat from '../Components/Photos/kat.jpg'
import adam from '../Components/Photos/adam.jpg'
import sam from '../Components/Photos/sam.jpg'
import cass from '../Components/Photos/cass.jpg'
import josh from '../Components/Photos/Josh.jpg'
import Navbar from './Navbar'

 function List(){
     return(
         <section className='List'>
         {exelist.map((exe) => {
         return <AboutUs exe={exe}></AboutUs>;
         })} 
         </section>
     )
 }

 const AboutUs = (props) =>{
     console.log(props)
 const {img, name, position}= props.exe;
 return(
 <article className='Exelist'>
    <img src={img} ></img>
     <h1>{name}</h1>
    <h1>{position}</h1>
 </article> 

 )
}
 export default List
