import { useState, useEffect } from 'react'
import Navigation from './components/Navigation'
import Hero from './sections/Hero'
import Problem from './sections/Problem'
import Solution from './sections/Solution'
import Features from './sections/Features'
import HowItWorks from './sections/HowItWorks'
import Demo from './sections/Demo'
import DemoVideos from './sections/DemoVideos'
import InteractivePlayground from './sections/InteractivePlayground'
import UseCases from './sections/UseCases'
import Stats from './sections/Stats'
import CTA from './sections/CTA'
import Footer from './sections/Footer'

function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light')

  useEffect(() => {
    // Check for saved theme preference or system preference
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null
    if (savedTheme) {
      setTheme(savedTheme)
      document.documentElement.setAttribute('data-theme', savedTheme)
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setTheme('dark')
      document.documentElement.setAttribute('data-theme', 'dark')
    }
  }, [])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
    localStorage.setItem('theme', newTheme)
  }

  return (
    <div className="app">
      <Navigation theme={theme} toggleTheme={toggleTheme} />
      <main>
        <Hero />
        <Problem />
        <Solution />
        <Features />
        <HowItWorks />
        <DemoVideos />
        <InteractivePlayground />
        <Demo />
        <UseCases />
        <Stats />
        <CTA />
      </main>
      <Footer />
    </div>
  )
}

export default App
