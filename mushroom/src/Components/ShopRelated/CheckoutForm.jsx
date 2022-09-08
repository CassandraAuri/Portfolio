import React, {useState, useContext} from 'react'
import { CardElement,useElements,useStripe } from '@stripe/react-stripe-js'
import { Context } from '../../Pages/Shop'
import axios from "axios"
export default function PaymentForm(props, context){ /*COPIED FROM DOCUMENTATION, NOT IMPLEMENTED*/
    const [success, setSuccess] =useState(false)
    const cart =useContext(props.context) /*Grabs the cart and quantites*/
    console.log(cart)
    const [prices, setPrices]=useState(0) /*Sets prices to 0*/
    const stripe = useStripe() 
    const elements = useElements()
    var Totalprice=0
    setPrices(Totalprice=cart.reduce((Total,Current)=>Total=100*(Total+Current.qty*Current.price),0))/*Changes price communicate with STRIPE*/
    const handleSubmit = async(e) =>{ /*Event*/
        e.preventDefault()
        const {error, paymentMethod} = await stripe.createPaymentMethod({
            type: "card",
            card:elements.getElement(CardElement)
        })
    
    if(!error){
        try {
            const {id}=paymentMethod
            const response = await axios.post("http://localhost:3000/checkout", {
            amount:{prices},
            id
            })
        if(response.data.success){
            console.log("successful payment")
            setSuccess(true)
        }
        } catch (error) {
            console.log("erorr", error)
        }
    }
        else{
            console.log(error.message)
        }
    }
    return(
        <div>
          TEST
        </div>
    )
}