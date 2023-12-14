// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Trainers__CommandLine__hxx__
#define __ivqML__Trainers__CommandLine__hxx__

#include <csignal>
#include <iostream>

// -------------------------------------------------------------------------
template< class _O > bool ivqML::Trainers::CommandLine< _O >::s_Stop = false;

// -------------------------------------------------------------------------
template< class _O >
ivqML::Trainers::CommandLine< _O >::
CommandLine( )
{
  // Detect ctrl-c event to stop optimization and finish training
  signal( SIGINT, []( int s ) -> void { Self::s_Stop = true; } );
  this->set_debug( Self::debug );
}

// -------------------------------------------------------------------------
template< class _O >
typename ivqML::Trainers::CommandLine< _O >::
TModel& ivqML::Trainers::CommandLine< _O >::
model( )
{
  return( this->m_Model );
}

// -------------------------------------------------------------------------
template< class _O >
const typename ivqML::Trainers::CommandLine< _O >::
TModel& ivqML::Trainers::CommandLine< _O >::
model( ) const
{
  return( this->m_Model );
}

// -------------------------------------------------------------------------
template< class _O >
bool ivqML::Trainers::CommandLine< _O >::
debug(
  const TScalar& J,
  const TScalar& G,
  const TModel* m,
  const TNatural& i,
  bool d
  )
{
  if( d )
    std::cerr << "J=" << J << ", Gn=" << G << ", i=" << i << std::endl;
  return( Self::s_Stop );
}

// -------------------------------------------------------------------------
template< class _O >
void ivqML::Trainers::CommandLine< _O >::
fit( )
{
  this->_prepare_training( );
  std::cerr << "Initial model: " << std::endl << this->m_Model << std::endl;
  this->init( this->m_Model, this->m_dX, this->m_dY );
  std::cerr << "-------------- START ---------------" << std::endl;
  this->Superclass::fit( );
  std::cerr << "-------------- FINISH --------------" << std::endl;
  std::cout << this->m_Model << std::endl;
}

#endif // __ivqML__Trainers__CommandLine__hxx__

// eof - $RCSfile$
