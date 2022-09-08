import React, {useContext, useState} from 'react'
import { Context } from './Shop'
import { loadStripe } from '@stripe/stripe-js'
import {Elements } from '@stripe/react-stripe-js'
import PaymentForm from '../Components/ShopRelated/CheckoutForm'
const PUBLIC_KEY="pk_test_51K1GdaLs0skxt0PPSFTDPrKjN5Gx9461ESVDUq3muwdA7xTh57eddcsa8TejbsK7wEJGPbklLxykVeaiPNsrMglz00V5MyoF9J"
const stripeTestPromise= loadStripe(PUBLIC_KEY)
const Checkout= ()=>{
    return(
        <div>
   <PaymentForm/>
     </div>
    )
    }
export default Checkout