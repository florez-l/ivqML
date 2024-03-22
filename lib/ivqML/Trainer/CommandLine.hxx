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
  } // end namespace
} // end namespace

// -------------------------------------------------------------------------
template< class _TOptimizer >
ivqML::Trainer::CommandLine< _TOptimizer >::
CommandLine( )
{
  signal( SIGINT, []( int s ) -> void { Self::s_Stop = true; } );
  this->set_debugger( Self::debugger );
}

// -------------------------------------------------------------------------
template< class _TOptimizer >
bool ivqML::Trainer::CommandLine< _TOptimizer >::
debugger(
  const TScl& J, const TScl& G,
  const TModel* m, const TNat& i, bool d
  )
{
  if( d )
  {
    std::cout
      << "iterations=" << i
      << " ; cost=" << J
      << " ; gradient_norm=" << G
      << std::endl;
  } // end if
  if( Self::s_Stop )
    std::cout << std::endl << std::endl;
  return( Self::s_Stop );
}

#endif // __ivqML__Trainer__CommandLine__hxx__

// eof - $RCSfile$
