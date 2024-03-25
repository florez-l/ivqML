// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Trainer__CommandLine__hxx__
#define __ivqML__Trainer__CommandLine__hxx__

#include <csignal>
#include <iostream>

namespace ivqML
{
  namespace Trainer
  {
    template< class _TOptimizer >
    bool CommandLine< _TOptimizer >::s_Stop = false;

    template< class _TOptimizer >
    typename _TOptimizer::TNat CommandLine< _TOptimizer >::m_DebugStep = 100;
  } // end namespace
} // end namespace

// -------------------------------------------------------------------------
template< class _TOptimizer >
ivqML::Trainer::CommandLine< _TOptimizer >::
CommandLine( )
{
  signal( SIGINT, []( int s ) -> void { Self::s_Stop = true; } );
  this->m_Debug = Self::debugger;
}

// -------------------------------------------------------------------------
template< class _TOptimizer >
void ivqML::Trainer::CommandLine< _TOptimizer >::
register_options( boost::program_options::options_description& opt )
{
  this->Superclass::register_options( opt );
  opt.add_options( )
    ivqML_Optimizer_OptionMacro( DebugStep, "debug_step" );
}

// -------------------------------------------------------------------------
template< class _TOptimizer >
bool ivqML::Trainer::CommandLine< _TOptimizer >::
debugger(
  const TModel* model,
  const TScl& norm, const TNat& iter,
  const TCost* cost,
  bool force
  )
{
  if( iter == 1 || iter % Self::m_DebugStep == 0 || force )
    std::cerr
      << "Iteration: " << iter
      << " | Gradient norm: " << norm
      << std::endl;
  return( Self::s_Stop );
}

#endif // __ivqML__Trainer__CommandLine__hxx__

// eof - $RCSfile$
